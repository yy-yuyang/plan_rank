import os
import tensorflow as tf
import tensorflow_ranking as tfr
from tensorflow_ranking.python import utils

import Fcnn

import numpy as np 

DFF_EN = 512           # 4 * D_MODEL 全连接第一层神经元数目
DFF_DN = 64
DROPOUT_RATE = 0.4
BUFFER_SIZE_TRAIN = 2000   #随机缓冲区大小
BUFFER_SIZE_TEST = 500
BATCHES_SIZE_TRAIN = 512      #一轮喂入数据样本数
BATCHES_SIZE_TEST = 256
EPOCHS = 500        #进行轮数

PATH_TRAIN = "../makedataset/train.npz"
PATH_TEST = "../makedataset/test.npz"




class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super().__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    #Optimizer
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def rank_accuracy(y_true, y_pred):
    # 获取 batch_size, num, _ 的维度
    shape = tf.shape(y_true)  # 返回的 shape 是一个 tensor，形状为 [batch, num, 1]
    batch_size = shape[0]
    num = shape[1]

    # 对 y_true 和 y_pred 按照每个 num 维度的值进行排序
    true_order = tf.argsort(tf.squeeze(y_true, axis=-1), axis=-1, direction='ASCENDING')  # [batch, num]
    pred_order = tf.argsort(tf.squeeze(y_pred, axis=-1), axis=-1, direction='ASCENDING')  # [batch, num]
    
    # 使用 tf.equal 来比较排序结果
    correct_order = tf.equal(true_order, pred_order)  # [batch, num]
    
    # 对每个样本进行统计，看每个 num 的排序是否正确
    correct_count = tf.reduce_sum(tf.cast(correct_order, tf.float32), axis=-1)  # [batch]
    
    # 计算准确率：每个样本的准确率除以 num
    accuracy = correct_count / tf.cast(num, tf.float32)  # [batch]

    # 返回平均准确率：对 batch 维度求均值
    return tf.reduce_mean(accuracy)


def main():
    data_train = np.load(PATH_TRAIN)
    train_input1 = data_train['tx']
    train_input2 = data_train['tz']
    train_output = data_train['ty']
    train_mask = data_train['mask']

    data_test = np.load(PATH_TEST)
    test_input1 = data_test['tx']
    test_input2 = data_test['tz']
    test_output = data_test['ty']
    test_mask = data_test['mask']

    train_dataset = tf.data.Dataset.from_tensor_slices(((train_input1, train_input2, train_mask), train_output))
    test_dataset = tf.data.Dataset.from_tensor_slices(((test_input1, test_input2, test_mask), test_output))

    train_dataset = train_dataset.shuffle(BUFFER_SIZE_TRAIN).batch(BATCHES_SIZE_TRAIN)

    test_dataset = test_dataset.shuffle(BUFFER_SIZE_TEST).batch(BATCHES_SIZE_TEST)


    learning_rate = CustomSchedule(256)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)    
    
    model = Fcnn.FCNN(dropout_rate = DROPOUT_RATE)
    
    model.compile(
        loss=mse_loss,
        optimizer=optimizer,
        metrics=[rank_accuracy])


    model.fit(
                    train_dataset,
                    epochs = EPOCHS,
                    verbose = 1,
                    validation_data = test_dataset)#callbacks = [ckpt_callback_save]
    #verbose 0:不输出信息；1:显示进度条(一般默认为1)；2:每个epoch输出一行记录

    model.save_weights("./model.h5", overwrite = True) 
    # model.evaluate()

    # model.summary()


if __name__=="__main__":                   
    main()
