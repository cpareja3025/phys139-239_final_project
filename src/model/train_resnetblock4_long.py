import random
import h5py
from pathlib import Path

import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Input, Conv2D, Reshape, Add, ReLU, Dropout,
    Flatten, Softmax, BatchNormalization, MaxPooling2D, AveragePooling2D
)

### set I/O path
project_dir = Path.cwd().parent.parent.resolve()

data_dir = project_dir.joinpath('data')
h5_dir = data_dir.joinpath('hdf5')
h5_train_path = h5_dir.joinpath('train.h5')
h5_test_path = h5_dir.joinpath('test.h5')

log_dir = project_dir.joinpath('logs').joinpath('resne4tblock_small_long_billy')
model_dir = project_dir.joinpath('models').joinpath('resnet4block_small_long')
best_dir = model_dir.joinpath('best')
latest_dir = model_dir.joinpath('latest')

csv_log_path = project_dir.joinpath('csv_logs').joinpath('resnet4block_small_long.csv')

################################ Start of network construction #########################################
################################# conv1  ##################################
resnet50_conv1_input = Input(shape=( 256, 256, 3), name='conv1_input' )

x = Conv2D(
    64, kernel_size=7, strides=2, activation='relu',
    padding='same', kernel_constraint=keras.constraints.max_norm(2.)
)(resnet50_conv1_input)

resnet50_conv1_output = MaxPooling2D(
    pool_size=3, padding='same', strides=2
)(x)



resnet50_conv1 = Model(
    inputs = resnet50_conv1_input,
    outputs = resnet50_conv1_output,
    name = "ResNet-50_conv_1block"
)


################################# conv2  ##################################
# First Block of conv2

resnet50_conv2_first_block_input = Input(shape=(64,64,64), name='conv2_first_block_input' )
x = Conv2D( 64, kernel_size=1, strides=1, activation='relu', padding='valid', kernel_constraint=keras.constraints.max_norm(2.))(resnet50_conv2_first_block_input)
x = BatchNormalization()(x)
x = Conv2D( 64, kernel_size=3, strides=1, activation='relu', padding='same', kernel_constraint=keras.constraints.max_norm(2.))(x)
x = BatchNormalization()(x)
x = Conv2D( 256, kernel_size=1, strides=1, padding='valid', kernel_constraint=keras.constraints.max_norm(2.))(x)
x = BatchNormalization()(x)

shortcut = Conv2D(
    256, kernel_size=1, strides=1, activation='relu', padding='same', kernel_constraint=keras.constraints.max_norm(2.)
)(resnet50_conv2_first_block_input)
shortcut = BatchNormalization()(shortcut)
x = Add()([x, shortcut])

resnet50_conv2_first_block_output = ReLU()(x)

resnet50_conv2_first_block = Model(
    inputs = resnet50_conv2_first_block_input,
    outputs = resnet50_conv2_first_block_output,
    name = 'resnet50_conv2_first_block'
)


# Identity Block of conv2
resnet50_conv2_identity_block_input = Input(shape=(64,64,256), name='conv2_identity_block_input' )
x = Conv2D( 64, kernel_size=1, strides=1, activation='relu', padding='valid', kernel_constraint=keras.constraints.max_norm(2.))(resnet50_conv2_identity_block_input)
x = BatchNormalization()(x)
x = Conv2D( 64, kernel_size=3, strides=1, activation='relu', padding='same', kernel_constraint=keras.constraints.max_norm(2.))(x)
x = BatchNormalization()(x)
x = Conv2D( 256, kernel_size=1, strides=1, activation='relu', padding='valid', kernel_constraint=keras.constraints.max_norm(2.))(x)
x = BatchNormalization()(x)

x = Add()([x, resnet50_conv2_identity_block_input ])
resnet50_conv2_identity_block_output = ReLU()(x)

resnet50_conv2_identity_block = Model(
    inputs = resnet50_conv2_identity_block_input,
    outputs= resnet50_conv2_identity_block_output,
    name = 'resnet50_conv2_identity_block'
)



# Combining the 2 types of blocks
resnet50_conv2_input = Input(shape=(64,64,64), name='resnet50_conv2_input')

x = resnet50_conv2_first_block(resnet50_conv2_input)
# x = resnet50_conv2_identity_block(resnet50_conv2_input)
x = resnet50_conv2_identity_block(x)
resnet50_conv2_output = resnet50_conv2_identity_block(x)

resnet50_conv2 = Model(
    inputs = resnet50_conv2_input,
    outputs = resnet50_conv2_output,
    name = "ResNet-50_conv2_block"
)


################################# conv3  ##################################
# First Block of conv3

resnet50_conv3_first_block_input = Input(shape=(64,64,256), name='conv3_first_block_input' )
x = Conv2D( 128, kernel_size=1, strides=2, activation='relu', padding='valid', kernel_constraint=keras.constraints.max_norm(2.))(resnet50_conv3_first_block_input)
x = BatchNormalization()(x)
x = Conv2D( 128, kernel_size=3, strides=1, activation='relu', padding='same', kernel_constraint=keras.constraints.max_norm(2.))(x)
x = BatchNormalization()(x)
x = Conv2D( 512, kernel_size=1, strides=1, padding='valid', kernel_constraint=keras.constraints.max_norm(2.))(x)
x = BatchNormalization()(x)

shortcut = Conv2D(512, kernel_size=1, strides=2, activation='relu', padding='same', kernel_constraint=keras.constraints.max_norm(2.))(resnet50_conv3_first_block_input)
shortcut = BatchNormalization()(shortcut)
x = Add()([ x, shortcut])
resnet50_conv3_first_block_output = ReLU()(x)

resnet50_conv3_first_block = Model(
    inputs  = resnet50_conv3_first_block_input,
    outputs = resnet50_conv3_first_block_output,
    name = 'resnet50_conv3_first_block'
)


# Identity Block of conv3
resnet50_conv3_identity_block_input = Input(shape=(32,32,512), name='conv3_identity_block_input' )
x = Conv2D( 128, kernel_size=1, strides=1, activation='relu', padding='valid', kernel_constraint=keras.constraints.max_norm(2.))(resnet50_conv3_identity_block_input)
x = BatchNormalization()(x)
x = Conv2D( 128, kernel_size=3, strides=1, activation='relu', padding='same', kernel_constraint=keras.constraints.max_norm(2.))(x)
x = BatchNormalization()(x)
x = Conv2D( 512, kernel_size=1, strides=1, activation='relu', padding='valid', kernel_constraint=keras.constraints.max_norm(2.))(x)
x = BatchNormalization()(x)

x = Add()([x, resnet50_conv3_identity_block_input ])
resnet50_conv3_identity_block_output = ReLU()(x)

resnet50_conv3_identity_block = Model(
    inputs = resnet50_conv3_identity_block_input,
    outputs= resnet50_conv3_identity_block_output,
    name = 'resnet50_conv3_identity_block'
)



# Combining the 2 types of blocks
resnet50_conv3_input = Input(shape=(64,64,256), name='resnet50_conv3_input' )

x = resnet50_conv3_first_block(resnet50_conv3_input)
x = resnet50_conv3_identity_block(x)
x = resnet50_conv3_identity_block(x)
resnet50_conv3_output = resnet50_conv3_identity_block(x)

resnet50_conv3 = Model(
    inputs = resnet50_conv3_input,
    outputs = resnet50_conv3_output,
    name = "ResNet-50_conv3_block"
)



################################# conv4  ##################################
# First Block of conv4

resnet50_conv4_first_block_input = Input(shape=(32,32,512), name='conv4_first_block_input' )
x = Conv2D( 256, kernel_size=1, strides=2, activation='relu', padding='valid', kernel_constraint=keras.constraints.max_norm(2.))(resnet50_conv4_first_block_input)
x = BatchNormalization()(x)
x = Conv2D( 256, kernel_size=3, strides=1, activation='relu', padding='same', kernel_constraint=keras.constraints.max_norm(2.))(x)
x = BatchNormalization()(x)
x = Conv2D( 1024, kernel_size=1, strides=1, padding='valid', kernel_constraint=keras.constraints.max_norm(2.))(x)
x = BatchNormalization()(x)

shortcut = Conv2D(1024, kernel_size=1, strides=2, activation='relu', padding='same', kernel_constraint=keras.constraints.max_norm(2.))(resnet50_conv4_first_block_input)
shortcut = BatchNormalization()(shortcut)
x = Add()([ x, shortcut])
resnet50_conv4_first_block_output = ReLU()(x)

resnet50_conv4_first_block = Model(
    inputs  = resnet50_conv4_first_block_input,
    outputs = resnet50_conv4_first_block_output,
    name = 'resnet50_conv4_first_block'
)



# Identity Block of conv4
resnet50_conv4_identity_block_input = Input(shape=(16,16,1024), name='conv4_identity_block_input' )
x = Conv2D( 256, kernel_size=1, strides=1, activation='relu', padding='valid', kernel_constraint=keras.constraints.max_norm(2.))(resnet50_conv4_identity_block_input)
x = BatchNormalization()(x)
x = Conv2D( 256, kernel_size=3, strides=1, activation='relu', padding='same', kernel_constraint=keras.constraints.max_norm(2.))(x)
x = BatchNormalization()(x)
x = Conv2D( 1024, kernel_size=1, strides=1, activation='relu', padding='valid', kernel_constraint=keras.constraints.max_norm(2.))(x)
x = BatchNormalization()(x)

x = Add()([x, resnet50_conv4_identity_block_input ])
resnet50_conv4_identity_block_output = ReLU()(x)

resnet50_conv4_identity_block = Model(
    inputs = resnet50_conv4_identity_block_input,
    outputs= resnet50_conv4_identity_block_output,
    name = 'resnet50_conv4_identity_block'
)




# Combining the 2 types of blocks
resnet50_conv4_input = Input(shape=(32,32,512), name='resnet50_conv4_input' )

x = resnet50_conv4_first_block(resnet50_conv4_input)
x = resnet50_conv4_identity_block(x)
x = resnet50_conv4_identity_block(x)
x = resnet50_conv4_identity_block(x)
x = resnet50_conv4_identity_block(x)
resnet50_conv4_output = resnet50_conv4_identity_block(x)

resnet50_conv4 = Model(
    inputs = resnet50_conv4_input,
    outputs = resnet50_conv4_output,
    name = "ResNet-50_conv4_block"
)

################################# classifier  ##################################
# The classifier will tell us whether this event is
# electron CC / muon CC / tauon CC / Neutral (4 types)


resnet50_classifier_input = Input(shape=(16,16,1024), name='classification')
x = AveragePooling2D(pool_size=2, padding='same')(resnet50_classifier_input)
x = Dropout(0.2)(x)
x = Flatten()(x)


# I think it's 4, but if we want to classify more types of events
# Need to change 4 to something else...
number_of_categories = 5

resnet50_classifier_output = Dense(
    number_of_categories, activation='softmax', kernel_constraint=keras.constraints.max_norm(2.)
)(x)


resnet50_classifier = Model(
    inputs = resnet50_classifier_input,
    outputs= resnet50_classifier_output,
    name = "ResNet-50_Classifier"
)



####################################################################################
def build_resnet50(optimizer, loss, metrics):
    resnet50_input = Input(shape=( 256, 256, 3))

    # image Augumentation ??
    x = resnet50_input

    x = resnet50_conv1(x)  # do conv in this block
    x = resnet50_conv2(x)
    x = resnet50_conv3(x)
    x = resnet50_conv4(x)

    resnet50_output = resnet50_classifier(x)

    resnet50_model = Model(
        inputs  = resnet50_input,
        outputs = resnet50_output,
        name = 'ResNet-50_Whole_Network'
    )

    print( resnet50_model.summary() )

    resnet50_model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )

    return resnet50_model

################################ End of network construction #########################################



# tf.keras.utils.plot_model(resnet_obj, show_shapes=True, show_dtype=True)

### define optimizerm, loss, metrics
loss = keras.losses.CategoricalCrossentropy(from_logits=False)
optimizer = keras.optimizers.Adam(learning_rate=0.0005)



ca = keras.metrics.CategoricalAccuracy(
    name='categorical_accuracy', dtype=None
)
metrics = [ca]

### build the resnet model and compile
resnet50_network = build_resnet50(loss=loss, optimizer=optimizer, metrics=metrics)


class generator:
    def __init__(self, file, mode, batch_size):
        self.file = file
        self.mode = mode

        self.batch_size = batch_size
        self.length = self.compute_length()

        self.indices = None

        self.avg = 0.37174486416699054
        self.std = 4439.282558540287

    def __call__(self):
        self.indices = list(range(self.length))
        random.shuffle(self.indices)

        with h5py.File(self.file, 'r') as hf:
            for i in range(int(self.length/self.batch_size)-1):
                sel_indices = [self.indices.pop() for _ in range(self.batch_size)]
                sel_indices.sort()

                sel_imgs = hf[f"X_{self.mode}"][sel_indices]
                sel_labels = hf[f"y_{self.mode}"][sel_indices]

                sel_imgs = sel_imgs.swapaxes(1,-1)
                sel_imgs = (sel_imgs-self.avg)/self.std
                #sel_labels = sel_labels.reshape(self.batch_size, 5)

                yield sel_imgs, sel_labels

    def compute_length(self):
        length = 0
        with h5py.File(self.file, 'r') as hf:
            length = len(hf[f"X_{self.mode}"])
        return length

batch_size=64
train_gen = generator(h5_train_path, mode='train', batch_size=batch_size)
test_gen = generator(h5_test_path, mode='test', batch_size=batch_size)

ds_train = tf.data.Dataset.from_generator(
    train_gen,
    output_signature=(
         tf.TensorSpec(shape=(batch_size, 256, 256, 3), dtype=tf.float32),
         tf.TensorSpec(shape=(batch_size, 5), dtype=tf.float32)
    )
)

ds_val = tf.data.Dataset.from_generator(
    test_gen,
    output_signature=(
         tf.TensorSpec(shape=(batch_size, 256, 256, 3), dtype=tf.float32),
         tf.TensorSpec(shape=(batch_size, 5), dtype=tf.float32)
    )
)

cb_tb = keras.callbacks.TensorBoard(log_dir=log_dir)
cb_csv = keras.callbacks.CSVLogger(csv_log_path)
cb_save_best = keras.callbacks.ModelCheckpoint(filepath=best_dir, monitor='val_loss', save_best_only=True)
cb_save_latest = keras.callbacks.ModelCheckpoint(filepath=latest_dir, monitor='val_loss', save_freq=10)

callbacks = [cb_tb, cb_csv, cb_save_best, cb_save_latest]

history = resnet50_network.fit(
    x=ds_train, epochs=500,
    validation_data=ds_val,
    callbacks=callbacks,
    workers=8,
    use_multiprocessing=True
)
