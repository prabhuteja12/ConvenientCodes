import os
import torch
from pytoune.framework.callbacks import Callback, PeriodicSaveCallback


osp = os.path


class TensorboardX(Callback):
    def __init__(self, writer, is_staggered=True):
        super(TensorboardX, self).__init__()
        self.writer = writer
        self.is_staggered = is_staggered
        if self.is_staggered:
            self.epoch_ind = 0
            self.batch_ind = 0

    def on_epoch_end(self, epoch, logs):
        if self.is_staggered:
            self.epoch_ind += 1
            epoch = self.epoch_ind

        self.writer.add_scalar('Train/loss_epoch', logs['loss'], epoch)

    def on_batch_end(self, batch, logs):
        if self.is_staggered:
            self.batch_ind += 1
            batch = self.batch_ind

        self.writer.add_scalar('Train/loss_batch', logs['loss'], batch)

        if batch % 100 == 0:
            for name, param in self.model.model.named_parameters():
                self.writer.add_histogram(name, param.clone().cpu().data.numpy(), batch)


class ModelCheckpointIter(Callback):
    def __init__(self, folder, n_iters=1000):
        super().__init__()
        self.savefolder = folder
        self.n_iters = n_iters
        self.curr_iter = 0
    
    def on_batch_end(self, batch, logs):
        net = self.model
        self.curr_iter += 1
        if self.curr_iter % self.n_iters == 0 and self.curr_iter != 0:
            torch.save(net, osp.join(self.savefolder, '{}_model.pth'.format(self.curr_iter)))


class LambdaCallback(Callback):
    def __init__(self, batch_end_cb=None, epoch_end_cb=None):
        super().__init__()
        self.be_cb = batch_end_cb
        self.ee_cb = epoch_end_cb
    
    def on_batch_end(self, batch, logs):
        # TODO
        if self.be_cb is not None:
            pass


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


class AdjustLrExponential(Callback):
    def __init__(self, base_lr, max_iter, power):
        super().__init__()
        self.base_lr = base_lr
        self.max_iter = max_iter
        self.power = power

    def on_batch_end(self, batch, logs):
        lr = lr_poly(self.base_lr, batch, self.max_iter, self.power)
        self.model.optimizer.param_groups[0]['lr'] = lr
        if len(self.model.optimizer.param_groups) > 1:
            self.model.optimizer.param_groups[1]['lr'] = lr * 10
