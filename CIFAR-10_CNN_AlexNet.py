# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = 'CPU'
import pandas as pd
from keras.models import Sequential
from tensorflow import keras
from keras.datasets import cifar10
from keras import backend as K
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, BatchNormalization, Activation, MaxPooling2D
from keras.models import Model
from keras.layers import concatenate,Dropout,Flatten
import matplotlib.pyplot as plt
from keras import optimizers,regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
import numpy as np

num_classes        = 10
batch_size         = 64         # 64 or 32 or other
epochs             = 90
iterations         = 782
DROPOUT=0.5 # keep 50%
CONCAT_AXIS=3
weight_decay=1e-4
DATA_FORMAT='channels_last' # Theano:'channels_first' Tensorflow:'channels_last'
log_filepath  = './alexnet'




def color_preprocessing(x_train,x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std  = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
        x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]
    return x_train, x_test

def scheduler(epoch):
    if epoch < 50:
        return 0.01
    if epoch < 100:
        return 0.001
    return 0.0001

# load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test  = keras.utils.to_categorical(y_test, num_classes)
x_train, x_test = color_preprocessing(x_train, x_test)





def alexnet(img_input):
    model = Sequential()
    model.add(Conv2D(28*2, (3, 3), input_shape=(32,32,3),strides=(1, 1), padding='same',kernel_regularizer=regularizers.l2(weight_decay),
               activation='relu',kernel_initializer='uniform')) # valid

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same',data_format=DATA_FORMAT))

    model.add(Conv2D(64*2, (5, 5), strides=(1, 1), padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same',data_format=DATA_FORMAT))

    model.add(Conv2D(96*2, (3, 3), strides=(1, 1), padding='same',
               activation='relu',kernel_initializer='uniform'))

    model.add(Conv2D(96*2, (3, 3), strides=(1, 1), padding='same',
               activation='relu',kernel_initializer='uniform'))

    model.add(Conv2D(64*2, (3, 3), strides=(1, 1), padding='same',
               activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same',data_format=DATA_FORMAT))
    model.add(Flatten())
    model.add(Dense(1024*2, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024*2, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model


img_input=Input(shape=(32,32,3))
# output = alexnet(img_input)
# model=Model(img_input,output)
model = alexnet(img_input)
model.summary()


# set optimizer
sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# set callback
tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
change_lr = LearningRateScheduler(scheduler)
cbks = [change_lr,tb_cb]

# set data augmentation
datagen = ImageDataGenerator(horizontal_flip=True,
                             width_shift_range=0.125,
                             height_shift_range=0.125,
                             fill_mode='constant',cval=0.)
datagen.fit(x_train)

# start training
history=model.fit(datagen.flow(x_train, y_train,batch_size=batch_size),
                    steps_per_epoch=iterations,
                    epochs=epochs,
                    callbacks=cbks,
                    validation_data=(x_test, y_test))

train_loss, train_acc = model.evaluate(x_train, y_train)
print('\nTrain accuracy: %.4f ' % train_acc)
test_loss, test_acc = model.evaluate(x_test, y_test)
print('\nTest accuracy: %.4f ' % test_acc)
print('\nPEG= %.4f ' % (1 - test_acc))

plt.subplot(211)
plt.title('CIFAR-10 Cross-Entropy Loss', pad=-40)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
# plot accuracy learning curves
plt.subplot(212)
plt.title('CIFAR-10 accuracy', pad=-40)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.legend()
plt.show()

# prediction=model.predict_classes(x_test)
# # prediction[:10]
# pd.crosstab(y_test.reshape(-1),prediction,rownames=['label'],colnames=['predict'])
# def DNNevaluate(model, Xtest, ytest):  # computes CM and PE for test set
#     Ntest = Xtest.shape[0] # number of rows
#     CM = np.zeros([10,10], dtype = int)
#     ypred = model.predict(Xtest) # predicts entire set
#
#     for i in range(Ntest):
#         yclass = np.argmax(ypred[i]) # index of the max element of y vector
#         ytrue = ytest[i]
#         CM[ytrue,yclass] += 1
#
#     Nerr = sum(sum(CM)) - np.trace(CM)
#     Ntotal = sum(sum(CM))
#     PEG = Nerr/Ntotal
#     return CM, PEG
#
# CM, PE = DNNevaluate(model,x_test,y_test)
# print('\nCM=\n',CM)
# print('\nPEG=', PE)