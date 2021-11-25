from keras import backend as K
import tensorflow as tf

def tversky(y_true, y_pred, smooth=1e-5, alpha=0.6):

    y_true = K.permute_dimensions(y_true, (3,1,2,0))
    y_pred = K.permute_dimensions(y_pred, (3,1,2,0))

    y_true_pos = K.batch_flatten(y_true)
    y_pred_pos = K.batch_flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos, 1)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos), 1)
    false_pos = K.sum((1-y_true_pos)*y_pred_pos, 1)
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return K.sum(1 - tversky(y_true, y_pred))


def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    tv = tversky(y_true, y_pred)
    return 1/7 * K.sum(K.pow((1 - tv), gamma))

def categorical_crossentropy(y_true, y_pred):    
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    cross_entropy = K.log(y_pred + 1e-5) * y_true
    return - K.sum(cross_entropy)
     
    
    