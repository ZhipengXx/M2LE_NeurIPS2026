import torch
import torch.nn as nn
from torch.nn import functional as F
import scipy.ndimage
import numpy as np

class AngularPenaltySMLoss(nn.Module):
    def __init__(self, loss_type='cosface', eps=1e-7, s=20, m=0):
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in ['arcface', 'sphereface', 'cosface', 'crossentropy']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 20.0 if not s else s
            self.m = 0.0 if not m else m
        if loss_type == 'crossentropy':
            self.s = 30.0 if not s else s
            self.m = 0.0 if not m else m
        self.loss_type = loss_type
        self.eps = eps

        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, wf, labels):
        if self.loss_type == 'crossentropy':
            return self.cross_entropy(wf, labels)
        else:
            if self.loss_type == 'cosface':
                numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
            if self.loss_type == 'arcface':
                numerator = self.s * torch.cos(torch.acos(
                    torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)) + self.m)
            if self.loss_type == 'sphereface':
                numerator = self.s * torch.cos(self.m * torch.acos(
                    torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)))

            excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
            denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
            L = numerator - torch.log(denominator)
            return -torch.mean(L)


class MahalanobisLoss(nn.Module):
    def __init__(self, sigma_old):

        super(MahalanobisLoss, self).__init__()
        self.sigma_inv = sigma_old

    def compute_mahalanobis_distance(self, embedding1, embedding2, cov_inv):

        embedding1 = F.normalize(embedding1, p=2, dim=0)
        embedding2 = F.normalize(embedding2, p=2, dim=0)

        delta = embedding1 - embedding2
        cov_inv = cov_inv + 1e-6 * torch.eye(cov_inv.size(0), device=cov_inv.device)
        intermediate = torch.matmul(cov_inv, delta.unsqueeze(1))
        distance = torch.matmul(delta.unsqueeze(0), intermediate)
        distance = torch.sqrt(distance + 1e-8).squeeze()
        return distance

    def forward(self, x, y, labels):
        unique_labels = torch.unique(labels)
        loss = 0.0

        for label in unique_labels:

            class_mask = (labels == label)
            x_class = x[class_mask]
            y_class = y[class_mask]
            cov_inv = self.sigma_inv[label.item()]

            for i in range(x_class.size(0)):
                for j in range(i + 1, x_class.size(0)):
                    d_old = self.compute_mahalanobis_distance(x_class[i], x_class[j], cov_inv)
                    d_new = self.compute_mahalanobis_distance(y_class[i], y_class[j], cov_inv)
                    loss += (d_old - d_new).abs()

        return loss / x.size(0)

def compute_angle_weighted_patch_distillation_loss(p_n, p_o, cls_n):

    p_n_normalized = F.normalize(p_n, p=2, dim=-1)  # [batch_size, num_tokens, embed_dim]
    p_o_normalized = F.normalize(p_o, p=2, dim=-1)  # [batch_size, num_tokens, embed_dim]

    alpha_cos = F.cosine_similarity(cls_n.unsqueeze(1),p_n,dim=-1) # [batch_size, num_tokens]
    alpha_cos = alpha_cos.clamp(min=-1.0, max=1.0)
    alpha_angle = 1 - (torch.acos(alpha_cos) / torch.pi)  # [batch_size, num_tokens]

    distances = torch.norm(p_n_normalized - p_o_normalized, p=2, dim=-1)  # [batch_size, num_tokens]
    weighted_distances = (1-alpha_angle.detach()) * distances  # [batch_size, num_tokens]
    loss = weighted_distances.mean()

    return loss


def _adaptive_threshold( svals: torch.Tensor, offset: float = 0):
    points: np.ndarray = svals.detach().cpu().numpy()
    assert points.ndim == 1
    if len(points) >= 128:
        fil_points = scipy.ndimage.gaussian_filter1d(points, sigma=10)
        _delta = 1
        diff_o1 = fil_points[:-_delta] - fil_points[_delta:]
        diff_o2 = diff_o1[:-1] - diff_o1[1:]
        _drop_ratio = 0.03
        drop_num = int(len(points) * _drop_ratio / 2)
        assert len(points) - drop_num >= 10
        valid_o2 = diff_o2[drop_num:-drop_num]
        thres_val = points[np.argmax(valid_o2) + int((len(points) - len(valid_o2)) / 2)]
    else:
        diff_o1 = points[:-1] - points[1:]
        diff_o2 = diff_o1[:-1] - diff_o1[1:]
        thres_val = points[np.argmax(diff_o2) + int((len(points) - len(diff_o2)) / 2)]
    i_thres = np.arange(len(points))[points >= thres_val].max()
    if 0 <= offset < 1:
        i_thres = min(i_thres + int(offset * (len(points) - i_thres)), len(points) - 1)
    else:
        i_thres = max(min(i_thres + int(offset), len(points) - 1), 0)

    zero_idx = np.zeros(len(points), dtype=np.int64)
    zero_idx[i_thres:] = 1
    zero_idx = torch.as_tensor(torch.from_numpy(zero_idx), dtype=torch.bool, device=svals.device)
    return ~zero_idx  ##返回的主成分为true

def _select_adaptive_principal_mask(svals: torch.Tensor) -> torch.Tensor:
    """
    Reuse Attention_LoRA._adaptive_threshold(offset=0) to select principal directions.
    Falls back to the top-1 direction when adaptive selection is unavailable or empty.
    """
    assert svals.ndim == 1
    if svals.numel() == 0:
        return torch.zeros_like(svals, dtype=torch.bool)

    mask = None
    try:
        # Keep the existing project logic as the single source of adaptive thresholding.
        mask = _adaptive_threshold( svals, offset=0)
    except Exception:
        mask = None

    if mask is None or mask.numel() != svals.numel():
        mask = torch.zeros_like(svals, dtype=torch.bool)
    else:
        mask = mask.to(device=svals.device, dtype=torch.bool)

    if int(mask.sum().item()) == 0:
        mask = torch.zeros_like(svals, dtype=torch.bool)
        mask[0] = True
    return mask


def compute_bdr_loss_branch(
    y_cur: torch.Tensor,
    y_base: torch.Tensor,
    A_weight: torch.Tensor,
    B_weight: torch.Tensor,
    lam: float,
    eps: float = 1e-5,
    detach_B: bool = False,
) -> torch.Tensor:
    """
    Compute conflict-only BDR (CO-BDR) loss for one branch from LoRA-only outputs.

    Args:
        y_cur: [B, N, C], current task LoRA output.
        y_base: [B, N, C], historical LoRA output.
    Returns:
        Scalar tensor loss.
    """
    B, N, C = y_cur.shape
    M = B * N

    # Keep signature compatibility with previous BDR implementation.
    _ = (A_weight, B_weight, lam)

    Z = y_cur.reshape(M, C)
    P = y_base.reshape(M, C)


    # Same preprocessing as previous BDR: center each channel then L2-normalize by channel.
    Z = Z - Z.mean(dim=0, keepdim=True)
    P = P - P.mean(dim=0, keepdim=True)

    Z = Z / (torch.norm(Z, p=2, dim=0, keepdim=True) + eps)
    P = P / (torch.norm(P, p=2, dim=0, keepdim=True) + eps)

    # Build covariance from historical branch and decompose into principal directions.
    sigma = (P.transpose(0, 1) @ P) / float(M)
    sigma = 0.5 * (sigma + sigma.transpose(0, 1))

    # eigvals, eigvecs = torch.linalg.eigh(sigma)  # ascending
    # eigvals = torch.clamp(eigvals, min=0.0)
    # order = torch.argsort(eigvals, descending=True)
    # eigvals = eigvals[order]
    # eigvecs = eigvecs[:, order]

    U, S, _ = torch.linalg.svd(sigma, full_matrices=False)
    S = torch.clamp(S, min=0.0)

    principal_mask = _select_adaptive_principal_mask(S)
    U = U[:, principal_mask]  # [C, K]

    # CO-BDR: penalize only reverse-aligned components (alpha < 0) along selected history directions.
    p_proj = P @ U  # [M, K]
    z_proj = Z @ U  # [M, K]

    p_energy = (p_proj * p_proj).sum(dim=0)  # [K]
    alpha = (z_proj * p_proj).sum(dim=0) / (p_energy + eps)
    alpha_neg = torch.minimum(alpha, torch.zeros_like(alpha))

    # conflict_energy = (alpha_neg.square() * p_energy).sum()
    # total_energy = p_energy.sum()
    # loss = conflict_energy / (total_energy + eps)

    loss = (alpha_neg * alpha_neg * p_energy).sum()


    alpha_pos = torch.maximum(alpha, torch.zeros_like(alpha))
    pos_energy_per_dir = alpha_pos.square() * p_energy
    total_parallel_energy_per_dir = alpha.square() * p_energy

    K = int(U.shape[1])
    total_s_energy = S.sum() + eps
    selected_sval_energy_ratio = (S[principal_mask].sum() / total_s_energy).detach().item()

    neg_mask = alpha < 0
    pos_mask = alpha > 0
    neg_energy = loss.sum()
    pos_energy = pos_energy_per_dir.sum()
    total_parallel_energy = total_parallel_energy_per_dir.sum()

    stats = {
        'selected_rank': float(K),
        'selected_rank_ratio': float(K) / float(C) if C > 0 else 0.0,
        'selected_sval_energy_ratio': float(selected_sval_energy_ratio),
        'neg_dir_ratio': float(neg_mask.float().mean().detach().item()) if alpha.numel() > 0 else 0.0,
        'neg_energy_ratio': float((neg_energy / (total_parallel_energy + eps)).detach().item()) if alpha.numel() > 0 else 0.0,
        'pos_energy_ratio': float((pos_energy / (total_parallel_energy + eps)).detach().item()) if alpha.numel() > 0 else 0.0,
        'alpha_neg_mean': float((-alpha[neg_mask]).mean().detach().item()) if torch.any(neg_mask) else 0.0,
        'alpha_pos_mean': float(alpha[pos_mask].mean().detach().item()) if torch.any(pos_mask) else 0.0,
        'conflict_active': float(torch.any(neg_mask).detach().item()) if alpha.numel() > 0 else 0.0,
        'branch_loss': float(loss.detach().item()),
    }

    # Same-direction sharing (alpha >= 0) is not penalized.

    return loss, stats

def compute_bdr_loss_kv(
    y_cur_k: torch.Tensor,
    y_base_k: torch.Tensor,
    y_cur_v: torch.Tensor,
    y_base_v: torch.Tensor,
    A_k_weight: torch.Tensor,
    B_k_weight: torch.Tensor,
    A_v_weight: torch.Tensor,
    B_v_weight: torch.Tensor,
    lam: float,
    eps: float = 1e-5,
    detach_B: bool = False,
) -> torch.Tensor:
    """
    Sum BDR losses from k and v branches using cached LoRA outputs.
    """
    loss_k, stats_k = compute_bdr_loss_branch(
        y_cur=y_cur_k,
        y_base=y_base_k,
        A_weight=A_k_weight,
        B_weight=B_k_weight,
        lam=lam,
        eps=eps,
        detach_B=detach_B,
    )
    loss_v, stats_v = compute_bdr_loss_branch(
        y_cur=y_cur_v,
        y_base=y_base_v,
        A_weight=A_v_weight,
        B_weight=B_v_weight,
        lam=lam,
        eps=eps,
        detach_B=detach_B,
    )
    stats = {f'k_{k}': v for k, v in stats_k.items()}
    stats.update({f'v_{k}': v for k, v in stats_v.items()})
    stats['conflict_active'] = float(max(stats_k.get('conflict_active', 0.0), stats_v.get('conflict_active', 0.0)))
    stats['total_loss'] = float((loss_k + loss_v).detach().item())

    # loss_k = compute_hsic_delta_loss(
    #     delta_t=y_cur_k[:, 0, :],
    #     delta_prev=y_base_k[:, 0, :],
    # )
    # loss_v = compute_hsic_delta_loss(
    #     delta_t=y_cur_v[:, 0, :],
    #     delta_prev=y_base_v[:, 0, :],
    # )
    return loss_k + loss_v,  stats


def pairwise_distances(x: torch.Tensor) -> torch.Tensor:
    # x should be two dimensional
    instances_norm = torch.sum(x ** 2, -1).reshape((-1, 1))
    return -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()


def GaussianKernelMatrix(x: torch.Tensor, sigma: float = 1) -> torch.Tensor:
    pairwise_distances_ = pairwise_distances(x)
    return torch.exp(-pairwise_distances_ / sigma)


def hsic(x: torch.Tensor, y: torch.Tensor, s_x: float = 1, s_y: float = 1) -> torch.Tensor:
    m, _ = x.shape  # batch size
    K = GaussianKernelMatrix(x, s_x)
    L = GaussianKernelMatrix(y, s_y)
    H = torch.eye(m) - 1.0 / m * torch.ones((m, m))
    H = H.to(x.device)
    HSIC = torch.trace(torch.mm(L, torch.mm(H, torch.mm(K, H)))) / ((m - 1) ** 2)
    return HSIC


def compute_hsic_delta_loss(delta_t: torch.Tensor, delta_prev: torch.Tensor) -> torch.Tensor:
    # Keep LEAR default bandwidths: sx=1, sy=1.
    return hsic(delta_t, delta_prev, s_x=1, s_y=1)
