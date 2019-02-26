
import time
import keras


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        elapsed = time.time() - self.epoch_time_start
        print("Elapsed time:", elapsed)
        self.times.append(elapsed)
