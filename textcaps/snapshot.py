import keras.callbacks as callbacks
from keras.callbacks import Callback
import numpy as np
import os

class SnapshotModelCheckpoint(Callback):
    def __init__(self, nb_epochs, nb_snapshots, fn_prefix='Model'):
        super(SnapshotModelCheckpoint, self).__init__()

        self.check = nb_epochs // nb_snapshots
        self.fn_prefix = fn_prefix

    def on_epoch_end(self, epoch, logs={}):
        if epoch != 0 and (epoch + 1) % self.check == 0:
            filepath = self.fn_prefix + "-%d.h5" % ((epoch + 1) // self.check)
            self.model.save(filepath, overwrite=True)


class SnapshotCallbackBuilder:
    def __init__(self, nb_epochs, nb_snapshots, init_lr, save_dir):
        self.T = nb_epochs
        self.M = nb_snapshots
        self.alpha_zero = init_lr
        self.save_dir = save_dir

    def get_callbacks(self,log, model_prefix='Model'):
        if not os.path.exists(self.save_dir+'/weights/'):
            os.makedirs(self.save_dir+'/weights/')

        callback_list = [callbacks.ModelCheckpoint(self.save_dir+"/weights/weights_{epoch:002d}.h5", monitor="val_capsnet_acc",
                                                    save_best_only=True, save_weights_only=False),
                         callbacks.LearningRateScheduler(schedule=self._cosine_anneal_schedule),
                         SnapshotModelCheckpoint(self.T, self.M, fn_prefix=self.save_dir+'/weights/%s' % model_prefix), log]

        return callback_list

    def _cosine_anneal_schedule(self, t):
        cos_inner = np.pi * (t % (self.T // self.M))  # t - 1 is used when t has 1-based indexing.
        cos_inner /= self.T // self.M
        cos_out = np.cos(cos_inner) + 1
        return float(self.alpha_zero / 2 * cos_out)