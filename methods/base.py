"""Shared task bookkeeping and evaluation for continual learning."""

class BaseLearner:
    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._device = args['device'][0]

    def eval_task(self):
        y_pred, y_pred_with_task, y_true, y_pred_task, y_true_task = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true, accuracy_matrix=True)
        cnn_accy_with_task = self._evaluate(y_pred_with_task, y_true)
        cnn_accy_task = (y_pred_task == y_true_task).sum().item() / len(y_pred_task)
        return cnn_accy, cnn_accy_with_task, cnn_accy_task
