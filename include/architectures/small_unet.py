from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, Input, MaxPooling2D, Concatenate, AveragePooling1D, Reshape, Activation, add, Conv2DTranspose, BatchNormalization, UpSampling2D, SeparableConv2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import plot_model, Sequence
from tensorflow.keras.losses import CategoricalCrossentropy, Reduction, BinaryCrossentropy
from tensorflow.keras.layers import Lambda
from tensorflow.keras.activations import softmax
from tensorflow import roll, norm, keras
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.applications as A

import loss

'''
small_unet
'''
def unet1(name, input_height, input_width, number_classes, metrics = None):
    input_l = Input(shape=(input_height,input_width,3))
    
    skip_layer_anchors = []
    
    filter_arr = [20,40,80,160]
    
    x = input_l
    
    for filters in filter_arr:
        x = Conv2D(filters, 3, strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        
        x = Dropout(0.3)(x)

        x = Conv2D(filters, 3, strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        
        x = Dropout(0.3)(x)
        
        skip_layer_anchors.append(x)
        
        x = MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding="same")(x)
        
    skip_layer_anchors.reverse()
    filter_arr.reverse()
    pool = Conv2D(320, (1, 1))(x)
    x = Activation("relu")(pool)

    for i,f in enumerate(filter_arr):
        x = Conv2D(f, 3, strides=1, padding='same')(x)
        x = UpSampling2D(interpolation='bilinear')(x)
        x = Activation("relu")(x)
        
        x = Concatenate()([skip_layer_anchors[i], x])

        x = Conv2D(f, 3, strides=1, padding='same')(x)
        x = Activation("relu")(x)
        x = Dropout(0.3)(x)
        x = Conv2D(f, 3, strides=1, padding='same')(x)
        x = Activation("relu")(x)
        x = Dropout(0.3)(x)
        
    output = Conv2D(number_classes, 1)(x)
    output = Activation('softmax')(output)
    
    model = Model(inputs=input_l, outputs=output, name=name)
    optimizer = Adam(lr=2e-3) # lr is learning rate
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics) # mean squared error because it is a regression problem
    #plot_model(model, to_file='%s.png' % (name))
    return model