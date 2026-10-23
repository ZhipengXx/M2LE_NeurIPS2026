"""M2LE: HIM merging and task simulation perturbation (TSP)."""

import logging
import re

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from methods.base import BaseLearner
from models.network import M2LENet
from models.attention import Attention_LoRA
from models.losses import AngularPenaltySMLoss
from utils.schedulers import CosineSchedule
from utils.toolkit import tensor2numpy, count_parameters


class Learner(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        self._network = M2LENet(args)
        for module in self._network.modules():
            if isinstance(module, Attention_LoRA):
                module.init_param()

        self.args = args
        self.optim = args["optim"]
        self.init_epoch = args["init_epoch"]
        self.init_lr = args["init_lr"]
        self.init_weight_decay = args["init_weight_decay"]
        self.epochs = args["epochs"]
        self.lrate = args["lrate"]
        self.batch_size = args["batch_size"]
        self.weight_decay = args["weight_decay"]
        self.num_workers = args["num_workers"]
        self.scale = args["scale"]
        self.margin = args["margin"]
        self.total_sessions = args["total_sessions"]


        self.use_meta = args["use_meta"]

        self.rho = args["rho"]
        self.beta = args["beta"]


        self.topk = 1
        self.class_num = self._network.class_num

        self.acc_matrix = np.zeros((self.total_sessions, self.total_sessions))

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):

        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)

        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train', mode='train')
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_workers, pin_memory=True)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                      num_workers=self.num_workers, pin_memory=True)

        for m in self._network.modules():
            if isinstance(m, Attention_LoRA):
                m.start_task()


        self._train(self.train_loader, self.test_loader)


        if self.args['use_merge'] is True:
            self._run_lora_merge_post_train()

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        network_params = []
        for name, param in self._network.named_parameters():
            param.requires_grad_(False)
            if re.search(rf"(^|\.)classifier_pool\.{self._network.numtask - 1}($|\.)", name) is not None:
                param.requires_grad_(True)
                network_params.append({'params': param})
            if re.search(rf"(^|\.)lora_B_k\.{self._network.numtask - 1}($|\.)", name) is not None:
                param.requires_grad_(True)
                network_params.append({'params': param})
            if re.search(rf"(^|\.)lora_B_v\.{self._network.numtask - 1}($|\.)", name) is not None:
                param.requires_grad_(True)
                network_params.append({'params': param})
            if re.search(rf"(^|\.)lora_A_k\.{self._network.numtask - 1}($|\.)", name) is not None:
                param.requires_grad_(True)
                network_params.append({'params': param})
            if re.search(rf"(^|\.)lora_A_v\.{self._network.numtask - 1}($|\.)", name) is not None:
                param.requires_grad_(True)
                network_params.append({'params': param})


        if self._cur_task==0:
            if self.optim == 'sgd':
                optimizer = optim.SGD(params=network_params, momentum=0.9,lr=self.init_lr,weight_decay=self.init_weight_decay)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=self.init_epoch)
            elif self.optim == 'adam':
                optimizer = optim.Adam(params=network_params,lr=self.init_lr,weight_decay=self.init_weight_decay, betas=(0.9,0.999))
                scheduler = CosineSchedule(optimizer=optimizer,K=self.init_epoch)
            else:
                raise Exception
            self.run_epoch = self.init_epoch
            self.train_function(train_loader,test_loader,optimizer,scheduler)
        else:
            if self.optim == 'sgd':
                optimizer = optim.SGD(params=network_params, momentum=0.9,lr=self.lrate,weight_decay=self.weight_decay)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=self.epochs)
            elif self.optim == 'adam':
                optimizer = optim.Adam(params=network_params,lr=self.lrate,weight_decay=self.weight_decay, betas=(0.9,0.999))
                scheduler = CosineSchedule(optimizer=optimizer,K=self.epochs)
            else:
                raise Exception
            self.run_epoch = self.epochs
            self.train_function(train_loader, test_loader, optimizer, scheduler)

    def train_function(self, train_loader, test_loader, optimizer, scheduler):
        logging.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        # Double check
        enabled = set()
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        logging.info(f"Parameters to be updated: {enabled}")

        prog_bar = tqdm(range(self.run_epoch))

        loss_cos = AngularPenaltySMLoss(loss_type='cosface',s=self.scale,m=self.margin)
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.


            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                mask = (targets >= self._known_classes).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, mask)
                targets = torch.index_select(targets, 0, mask)-self._known_classes


                if  not self.use_meta or torch.unique(targets).shape[0] == 1:
                    output = self._network(inputs)
                    logits = output['logits']
                    loss=loss_cos(logits, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                else:

                    tsp_loss, logits,targets = self._tsp_step(inputs, targets, loss_cos,optimizer)

                    loss =   tsp_loss

                losses += loss.item()


                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)


            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task,
                epoch + 1,
                self.run_epoch,
                losses / len(train_loader),
                train_acc
            )
            prog_bar.set_description(info)


        # task train finished
        test_acc = self._compute_accuracy(self._network, test_loader)
        final_info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task,
                epoch + 1,
                self.run_epoch,
                losses / len(train_loader),
                train_acc,
                test_acc,
            )
        logging.info(final_info)

    def accuracy(self, y_pred, y_true, accuracy_matrix = False):
        assert len(y_pred) == len(y_true), 'Data length error.'
        all_acc = {}
        all_acc['total'] = np.around((y_pred == y_true).sum()*100 / len(y_true), decimals=2)

        i = 0
        # Grouped accuracy
        for class_id in range(0, np.max(y_true), self.class_num):
            idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + self.class_num))[0]
            label = '{}-{}'.format(str(class_id).rjust(2, '0'), str(class_id+self.class_num-1).rjust(2, '0'))
            all_acc[label] = np.around((y_pred[idxes] == y_true[idxes]).sum()*100 / len(idxes), decimals=2)
            if accuracy_matrix:
                self.acc_matrix[i, self._cur_task] = all_acc[label]
            i += 1

        # Old accuracy
        idxes = np.where(y_true < self._known_classes)[0]
        all_acc['old'] = 0 if len(idxes) == 0 else np.around((y_pred[idxes] == y_true[idxes]).sum()*100 / len(idxes),
                                                            decimals=2)

        # New accuracy
        idxes = np.where(y_true >= self._known_classes)[0]
        all_acc['new'] = np.around((y_pred[idxes] == y_true[idxes]).sum()*100 / len(idxes), decimals=2)

        return all_acc

    def _evaluate(self, y_pred, y_true, accuracy_matrix=False):
        ret = {}
        grouped = self.accuracy(y_pred, y_true, accuracy_matrix=accuracy_matrix)
        ret['grouped'] = grouped
        ret['top1'] = grouped['total']
        return ret

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        y_pred_with_task = []
        y_pred_task, y_true_task = [], []

        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)

            with torch.no_grad():
                task_id = (targets//self.class_num).cpu()
                y_true_task.append(task_id)

                outputs = self._network.interface(inputs)

            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1].view(-1)  # [bs, topk]
            y_pred_task.append((predicts//self.class_num).cpu())

            outputs_with_task = torch.zeros_like(outputs)[:,:self.class_num]
            for idx, i in enumerate(targets//self.class_num):
                en, be = self.class_num*i, self.class_num*(i+1)
                outputs_with_task[idx] = outputs[idx, en:be]
            predicts_with_task = outputs_with_task.argmax(dim=1)
            predicts_with_task = predicts_with_task + (targets//self.class_num)*self.class_num

            y_pred.append(predicts.cpu().numpy())
            y_pred_with_task.append(predicts_with_task.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_pred_with_task), np.concatenate(y_true), torch.cat(y_pred_task), torch.cat(y_true_task)  # [N, topk]

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model.interface(inputs)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct)*100 / total, decimals=2)

    def _get_lora_modules(self):
        return [m for m in self._network.modules() if isinstance(m, Attention_LoRA)]

    @torch.no_grad()
    def _collect_lora_delta_map(self, lora_modules):
        delta_map = {}
        for m in lora_modules:
            delta_k, delta_v = m.get_matrix(self._cur_task)
            delta_map[m] = (delta_k.detach().float(), delta_v.detach().float())
        return delta_map

    def _run_lora_merge_post_train(self):
        """Estimate current-task Fisher before merging, then update HIM statistics."""
        lora_modules = self._get_lora_modules()
        if len(lora_modules) == 0:
            return

        ridge_lambda = float(self.args.get("ridge_lambda", 1e-9))
        empFI = bool(self.args.get("empFI", True))
        max_fisher_batches = self.args.get("fisher_max_batches", None)
        if max_fisher_batches is not None:
            max_fisher_batches = int(max_fisher_batches)


        delta_map = self._collect_lora_delta_map(lora_modules)

        self._merge_lora_attention(lora_modules, delta_map, ridge_lambda, empFI, max_fisher_batches)  #H_i

    def _merge_lora_attention(self, lora_modules, delta_map, ridge_lambda, empFI, max_fisher_batches):
        fisher_map = {}
        for m in lora_modules:
            delta_k, delta_v = delta_map[m]
            fisher_map[m] = (
                torch.zeros_like(delta_k, dtype=torch.float32, device=delta_k.device),
                torch.zeros_like(delta_v, dtype=torch.float32, device=delta_v.device),
            )

        loss_fisher = AngularPenaltySMLoss(loss_type='cosface', s=self.scale, m=self.margin)
        self._network.eval()
        for m in lora_modules:
            m.collect_fisher = True

        denom_batches = len(self.train_loader)
        if max_fisher_batches is not None:
            denom_batches = min(denom_batches, max_fisher_batches)

        for batch_idx, (_, inputs, targets) in enumerate(self.train_loader):
            if max_fisher_batches is not None and batch_idx >= max_fisher_batches:
                break
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            mask = (targets >= self._known_classes).nonzero().view(-1)
            if mask.numel() == 0:
                continue
            inputs = torch.index_select(inputs, 0, mask)
            targets = torch.index_select(targets, 0, mask) - self._known_classes

            self._network.zero_grad(set_to_none=True)
            output = self._network(inputs)
            logits = output['logits']
            if not empFI:
                targets = torch.argmax(logits, dim=1)
            loss = loss_fisher(logits, targets)
            loss.backward()

            scale = inputs.size(0) / float(denom_batches)
            for m in lora_modules:
                if (
                    m._fisher_delta_k is None
                    or m._fisher_delta_v is None
                    or m._fisher_delta_k.grad is None
                    or m._fisher_delta_v.grad is None
                ):
                    continue
                grad_k = m._fisher_delta_k.grad.detach()
                grad_v = m._fisher_delta_v.grad.detach()
                fisher_map[m][0].add_(grad_k.float().pow(2) * scale)
                fisher_map[m][1].add_(grad_v.float().pow(2) * scale)

        for m in lora_modules:
            m.collect_fisher = False
            m._fisher_delta_k = None
            m._fisher_delta_v = None

        with torch.no_grad():
            for m in lora_modules:

                delta_k, delta_v = delta_map[m]
                H_t_k, H_t_v = fisher_map[m]

                A_prev_k = m.H_sum_k.clone()
                A_prev_v = m.H_sum_v.clone()

                if self._cur_task == 0:
                    theta_star_k = delta_k
                    theta_star_v = delta_v
                else:

                    A_prev_k_w = A_prev_k
                    A_prev_v_w = A_prev_v

                    alpha_k = float(self.args["alpha"])
                    alpha_v = float(self.args["alpha"])


                    H_t_k_w = H_t_k * alpha_k
                    H_t_v_w = H_t_v * alpha_v
                    denom_k = H_t_k_w + A_prev_k_w + ridge_lambda
                    denom_v = H_t_v_w + A_prev_v_w + ridge_lambda
                    numer_k = H_t_k_w * delta_k
                    numer_v = H_t_v_w * delta_v


                    theta_star_k = torch.where(denom_k != 0, numer_k / denom_k, torch.zeros_like(denom_k))
                    theta_star_v = torch.where(denom_v != 0, numer_v / denom_v, torch.zeros_like(denom_v))

                m.H_sum_k.add_(H_t_k)
                m.H_sum_v.add_(H_t_v)
                m.pre_k.add_(theta_star_k)
                m.pre_v.add_(theta_star_v)
                m.use_merged_weight = True

    def _split_label_disjoint_subsets(self, targets: torch.Tensor):
        unique_labels = torch.unique(targets)
        if unique_labels.numel() < 2:
            return None, None

        perm = torch.unique(targets)[torch.randperm(len(torch.unique(targets)), device=targets.device)]
        num_a = max(1, len(torch.unique(targets))//2)
        labels_a = perm[:num_a]
        labels_b = perm[num_a:]
        idx_a = torch.isin(targets, labels_a).nonzero(as_tuple=True)[0]
        idx_b = torch.isin(targets, labels_b).nonzero(as_tuple=True)[0]
        if idx_a.numel() == 0 or idx_b.numel() == 0:
            return None, None
        return idx_a, idx_b

    def _tsp_step(self, inputs, targets, loss_cos, optimizer):
        """Perform the two directional TSP updates on disjoint label subsets."""
        idx_a, idx_b = self._split_label_disjoint_subsets(targets)
        rho = float(self.rho) *torch.rand(1, device=self._device)

        probe_params = []
        cls_params = []

        for name, p in self._network.named_parameters():
            if not p.requires_grad:
                continue

            if (("lora_B_k" in name or "lora_B_v" in name) or
                ("lora_A_k" in name or "lora_A_v" in name)):
                probe_params.append(p)


            if re.search(rf"(^|\.)classifier_pool\.{self._network.numtask - 1}($|\.)", name) is not None:
                cls_params.append(p)

        lora_params = probe_params
        update_params = lora_params + cls_params
        n_lora = len(lora_params)


        def _add_grad_(params, grads):
            with torch.no_grad():
                for p, g in zip(params, grads):
                    if g is None:
                        continue
                    if p.grad is None:
                        p.grad = g.detach().clone()
                    else:
                        p.grad.add_(g.detach())

        def _look_ahead_loss(src_idx, dst_idx, beta: float = 1.0):
            src_inputs = inputs[src_idx]
            src_targets = targets[src_idx]
            dst_inputs = inputs[dst_idx]
            dst_targets = targets[dst_idx]


            src_logit = self._network(src_inputs)["logits"]

            src_grads = torch.autograd.grad(
                loss_cos(src_logit, src_targets), probe_params, create_graph=False, retain_graph=False
            )


            dst_logit_clean = self._network(dst_inputs)["logits"]
            dst_clean_loss = beta * (1.0 - self.beta) * loss_cos(dst_logit_clean, dst_targets)
            clean_grads = torch.autograd.grad(
                dst_clean_loss, update_params, create_graph=False, retain_graph=False
            )
            lora_clean_grads = clean_grads[:n_lora]
            cls_clean_grads = clean_grads[n_lora:]


            with torch.no_grad():
                flat_norm = [g.norm() + 1e-12 for g in src_grads]
                eps_list = [-rho * g / flat_norm[i] for i, g in enumerate(src_grads)]
                for p, e in zip(probe_params, eps_list):
                    p.add_(e)


            out = self._network(dst_inputs)["logits"]
            loss_pert = beta *self.beta*  loss_cos(out, dst_targets)

            optimizer.zero_grad()
            loss_pert.backward()


            _add_grad_(lora_params, lora_clean_grads)
            _add_grad_(cls_params, cls_clean_grads)

            optimizer.step()


            with torch.no_grad():
                for p, e in zip(probe_params, eps_list):
                    p.sub_(e)

            loss = loss_pert + dst_clean_loss

            return loss, out

        loss_a, out_a = _look_ahead_loss(idx_a, idx_b)
        loss_b, out_b = _look_ahead_loss(idx_b, idx_a)

        logits = torch.cat([out_b, out_a], dim=0)
        targets_out = torch.cat([targets[idx_a], targets[idx_b]], dim=0)
        tsp_loss = loss_a + loss_b
        return tsp_loss, logits, targets_out
