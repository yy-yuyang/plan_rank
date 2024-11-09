import numpy as np
import tensorflow as tf
import time
import random

import Dataprocess
import Fcnn
import Train


MODEL_PATH = "./model_mask.h5"

def LoadModel(modelpath):
    global model
    
    model = Fcnn.FCNN(dropout_rate = Train.DROPOUT_RATE)
    
    model.build(input_shape = [(None, 10, 64, 8), (None, 10, 192, 1)])
    
    model.load_weights(modelpath)


def Predict(planlist):
    global model
    
    x_plans, x_idxs = Dataprocess.Plans2Vectors(planlist)
    temp = model.call([x_plans, x_idxs])
    pred = tf.argmax(temp, axis = -1).numpy().reshape(-1)

    return pred



if __name__=="__main__":
    LoadModel()
    Predict()