from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, Input, MaxPooling2D, concatenate, AveragePooling1D, Reshape, Activation, add, Conv2DTranspose, BatchNormalization, UpSampling2D, SeparableConv2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import plot_model, Sequence
from tensorflow.keras.losses import CategoricalCrossentropy, Reduction, BinaryCrossentropy
from tensorflow.keras.layers import Lambda
from tensorflow.keras.activations import softmax
from tensorflow import roll, norm, keras
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.applications as A

import segmentation_models as sm # for simple segmentation architecture


def reference(name, input_height, input_width, number_classes):
   # base_model = A.ResNet50V2(include_top=False, weights="imagenet", input_shape=(input_height,input_width,3)) 
    #base_model = A.InceptionV3(include_top=False, weights="imagenet", input_shape=(input_height,input_width,3)) 
    base_model = A.VGG16(include_top=False, weights="imagenet", input_shape=(input_height,input_width,3))


    x = Conv2D(256, 1)(base_model.layers[-1].output)
    # previous_block_activation = x

    ### [Second half of the network: upsampling inputs] ###

    for filters in [128,128, 64, 64, 32]:
        x = Activation("relu")(x)
        x = Conv2DTranspose(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        # x = Activation("relu")(x)
        # x = Conv2DTranspose(filters, 3, padding="same")(x)
        # x = BatchNormalization()(x)

        x = UpSampling2D(2)(x)

        # # Project residual
        # residual = UpSampling2D(2)(previous_block_activation)
        # residual = Conv2D(filters, 1, padding="same")(residual)
        # x = add([x, residual])  # Add back residual
        # previous_block_activation = x  # Set aside next residual
        
    output = Conv2D(number_classes, 1)(x)
    output = Activation('softmax')(output)
    
    model = Model(inputs=base_model.inputs, outputs=output, name=name)
    optimizer = Adam(lr=0.01) # lr is learning rate
    loss = sm.losses.CategoricalCELoss() + sm.losses.DiceLoss()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[sm.metrics.iou_score]) # mean squared error because it is a regression problem
    #plot_model(model, to_file='%s.png' % (name))
    return model
def test_custom(name):
   # base_model = A.ResNet50V2(include_top=False, weights="imagenet", input_shape=(input_height,input_width,3)) 
   # base_model = A.InceptionV3(include_top=False, weights="imagenet", input_shape=(input_height,input_width,3)) 
    #base_model = A.VGG16(include_top=False, weights="imagenet", input_shape=(input_height,input_width,3))

    #inputs = Input(shape=(input_height,input_width,3))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128]:
        x = Activation("relu")(x)
        x = SeparableConv2D(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = Activation("relu")(x)
        x = SeparableConv2D(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [128, 64, 32]:
        x = Activation("relu")(x)
        x = Conv2DTranspose(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = Activation("relu")(x)
        x = Conv2DTranspose(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = UpSampling2D(2)(x)

        # Project residual
        residual = UpSampling2D(2)(previous_block_activation)
        residual = Conv2D(filters, 1, padding="same")(residual)
        x = add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual
        
    output = Conv2D(number_classes, 1)(x)
    output = Activation('softmax')(output)
    
    model = Model(inputs=inputs, outputs=output, name=name)
    optimizer = Adam(learnng_rate=1e-4) # lr is learning rate
    loss = sm.losses.CategoricalCELoss() + sm.losses.DiceLoss()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[sm.metrics.iou_score]) # mean squared error because it is a regression problem
    #plot_model(model, to_file='%s.png' % (name))
    return model