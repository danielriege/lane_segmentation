from keras import backend as K
import tensorflow as tf

# tensor like y_true = (B x H x W x C)

## Basic
def remove_background(tensor):
    # TODO: remove magic numbers
    return tf.slice(tensor, [0,0,0,0], [8, 224, 640, 6]) # remove last C

def tp(y_true, y_pred):
    y_true = remove_background(y_true)
    y_pred = remove_background(y_pred)
    return K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

def possible_positives(y_true, y_pred):
    y_true = remove_background(y_true)
    return K.sum(K.round(K.clip(y_true, 0, 1)))

def predicted_positives(y_true, y_pred):
    y_pred = remove_background(y_pred)
    return K.sum(K.round(K.clip(y_pred, 0, 1)))

############

def fn(y_true, y_pred):
    return possible_positives(y_true, y_pred) - tp(y_true, y_pred)

def fp(y_true, y_pred):
    return predicted_positives(y_true, y_pred) - tp(y_true, y_pred)

def recall(y_true, y_pred):
    return tp(y_true, y_pred) / (possible_positives(y_true, y_pred) + K.epsilon()) # add epsilon for zero divison prevention

def precision(y_true, y_pred):
    return tp(y_true, y_pred) / (predicted_positives(y_true, y_pred) + K.epsilon())

def f1_score(y_true, y_pred):
    precision_m = precision(y_true, y_pred)
    recall_m = recall(y_true, y_pred)
    return 2*((precision_m*recall_m)/(precision_m+recall_m+K.epsilon()))