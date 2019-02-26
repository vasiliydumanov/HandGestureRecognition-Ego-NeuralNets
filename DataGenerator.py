import numpy as np
import keras
from PIL import Image
from keras.utils import to_categorical


class DataGenerator(keras.utils.Sequence):
    def __init__(self, data, num_classes, batch_size=4):
        """Initialization"""
        self.data = data
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.indexes = np.arange(len(data))

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.data) // self.batch_size

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch = self.data[batch_indexes]
        images = np.array([np.array(Image.open(path)) for path in batch[:, 0]])
        labels = to_categorical(batch[:, 1], num_classes=self.num_classes)

        return images, labels

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.indexes)
