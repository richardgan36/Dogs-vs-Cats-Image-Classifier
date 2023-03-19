import tensorflow as tf
import numpy as np
import glob

from keras import layers, Input
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint

from plot_metrics import plot_graph
from generator_class import DataGenerator

NUM_IMAGES = 25000

# Hyperparameters
filters_conv0 = 64
filters_conv1 = 32
filters_conv2 = 32
dropout = 0.5
n_dense0 = 32
validation_split = 0.2
epochs = 30

class_params = {'dim': (50, 50),
                'batch_size': 64,
                'n_classes': 2,
                'n_channels': 1,
                'shuffle': True,
                'directory': '/Users/richardgan/Pictures/Machine Learning/train_dogVcats/'}


def main():
    partition = {'train': [], 'validation': []}
    labels = {}

    training_samples = np.ceil((1 - validation_split) * NUM_IMAGES)
    for i, filename in enumerate(glob.iglob('/Users/richardgan/Pictures/Machine Learning/train_dogVcats/*')):
        ID = filename[59:]  # Name of image without directory name
        if i < training_samples:
            partition['train'].append(ID)
        else:
            partition['validation'].append(ID)

        labels[ID] = 1 if ID.startswith('dog') else 0

    # Generators
    training_generator = DataGenerator(partition['train'], labels, **class_params)
    validation_generator = DataGenerator(partition['validation'], labels, **class_params)

    model = Sequential()
    model.add(Input(shape=(*class_params['dim'], class_params['n_channels'])))
    model.add(layers.Conv2D(filters_conv0, 5, activation='relu'))
    # model.add(layers.Conv2D(filters_conv0, 3, activation='relu'))
    model.add(layers.MaxPool2D())
    model.add(layers.Dropout(dropout))

    model.add(layers.Conv2D(filters_conv1, 5, activation='relu'))
    # model.add(layers.Conv2D(filters_conv1, 3, activation='relu'))
    model.add(layers.MaxPool2D())
    model.add(layers.Dropout(dropout))

    # model.add(layers.Conv2D(filters_conv2, 3, activation='relu'))
    # model.add(layers.MaxPool2D())
    # model.add(layers.Dropout(dropout))

    model.add(layers.Flatten())
    model.add(layers.Dense(n_dense0, activation='relu'))
    model.add(layers.Dropout(dropout))

    model.add(layers.Dense(2, activation='sigmoid'))  # Binary output layer

    # model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=3,
                                   verbose=1)

    model_checkpoint = ModelCheckpoint(filepath='trained_models/dogsVcats_model4.h5',
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True)

    history = model.fit(x=training_generator,
                        validation_data=validation_generator,
                        epochs=epochs,
                        callbacks=[early_stopping, model_checkpoint],
                        verbose=1,
                        use_multiprocessing=True,
                        workers=6)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epoch_range = np.arange(len(acc))
    plot_graph(epoch_range, acc, val_acc, loss, val_loss)


if __name__ == "__main__":
    main()
