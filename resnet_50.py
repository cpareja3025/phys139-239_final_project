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



################################# conv5  ################################## 
# First Block of conv5
resnet50_conv5_first_block_input = Input(shape=(16,16,1024), name='conv5_first_block_input' )
x = Conv2D( 512, kernel_size=1, strides=2, activation='relu', padding='valid', kernel_constraint=keras.constraints.max_norm(2.))(resnet50_conv5_first_block_input)
x = BatchNormalization()(x)
x = Conv2D( 512, kernel_size=3, strides=1, activation='relu', padding='same', kernel_constraint=keras.constraints.max_norm(2.))(x)
x = BatchNormalization()(x)
x = Conv2D( 2048, kernel_size=1, strides=1, padding='valid', kernel_constraint=keras.constraints.max_norm(2.))(x)
x = BatchNormalization()(x)

shortcut = Conv2D(2048, kernel_size=1, strides=2, activation='relu', padding='same', kernel_constraint=keras.constraints.max_norm(2.))(resnet50_conv5_first_block_input)
shortcut = BatchNormalization()(shortcut)
x = Add()([ x, shortcut])
resnet50_conv5_first_block_output = ReLU()(x)

resnet50_conv5_first_block = Model(
    inputs  = resnet50_conv5_first_block_input, 
    outputs = resnet50_conv5_first_block_output, 
    name = 'resnet50_conv5_first_block'
)




# Identity Block of conv5
resnet50_conv5_identity_block_input = Input(shape=(8,8,2048), name='conv5_identity_block_input' )
x = Conv2D( 512, kernel_size=1, strides=1, activation='relu', padding='valid', kernel_constraint=keras.constraints.max_norm(2.))(resnet50_conv5_identity_block_input)
x = BatchNormalization()(x)
x = Conv2D( 512, kernel_size=3, strides=1, activation='relu', padding='same', kernel_constraint=keras.constraints.max_norm(2.))(x)
x = BatchNormalization()(x)
x = Conv2D( 2048, kernel_size=1, strides=1, activation='relu', padding='valid', kernel_constraint=keras.constraints.max_norm(2.))(x)
x = BatchNormalization()(x)

x = Add()([x, resnet50_conv5_identity_block_input ])
resnet50_conv5_identity_block_output = ReLU()(x)

resnet50_conv5_identity_block = Model( 
    inputs = resnet50_conv5_identity_block_input, 
    outputs= resnet50_conv5_identity_block_output, 
    name = 'resnet50_conv5_identity_block'
)




# Combining the 2 types of blocks  
resnet50_conv5_input = Input(shape=(16,16,1024), name='resnet50_conv5_input' )

x = resnet50_conv5_first_block(resnet50_conv5_input)
x = resnet50_conv5_identity_block(x)
resnet50_conv5_output = resnet50_conv5_identity_block(x) 

resnet50_conv5 = Model(
    inputs = resnet50_conv5_input, 
    outputs = resnet50_conv5_output, 
    name = "ResNet-50_conv5_block"
)

# tf.keras.utils.plot_model(resnet50_conv5, show_shapes=True, show_dtype=True)

################################# classifier  ################################## 
# The classifier will tell us whether this event is 
# electron CC / muon CC / tauon CC / Neutral (4 types)


resnet50_classifier_input = Input(shape=(8,8,2048), name='classification')
x = AveragePooling2D(pool_size=2, padding='same')(resnet50_classifier_input)
x = Dropout(0.2)(x) 
x = Flatten()(x)


# I think it's 4, but if we want to classify more types of events 
# Need to change 4 to something else... 
number_of_categories = 4

resnet50_classifier_output = Dense(
    number_of_categories, activation='softmax', kernel_constraint=keras.constraints.max_norm(2.)
)(x)


resnet50_classifier = Model(
    inputs = resnet50_classifier_input, 
    outputs= resnet50_classifier_output, 
    name = "ResNet-50_Classifier"
)



#################################################################################### 
def build_resnet50(): 
    resnet50_input = Input(shape=( 256, 256, 3))
    
    # image Augumentation ?? 
    x = resnet50_input 
    
    x = resnet50_conv1(x)  # do conv in this block  
    x = resnet50_conv2(x)
    x = resnet50_conv3(x)
    x = resnet50_conv4(x)
    x = resnet50_conv5(x)
    
    resnet50_output = resnet50_classifier(x)
    
    resnet50_model = Model( 
        inputs  = resnet50_input, 
        outputs = resnet50_output, 
        name = 'ResNet-50_Whole_Network'
    )
    
    print( resnet50_model.summary() )
        
    resnet50_model.compile( 
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return resnet50_model

################################ End of network construction ######################################### 
    
resnet50_network = build_resnet50()

# tf.keras.utils.plot_model(resnet_obj, show_shapes=True, show_dtype=True)


random_inputs = np.random.rand(500, 256, 256, 3)   # Some random data 
random_labels = np.random.rand(500, 1, 1, 1)       # Some random data
 

history = resnet50_network.fit(
    random_inputs, random_labels, epochs=10
    
)


