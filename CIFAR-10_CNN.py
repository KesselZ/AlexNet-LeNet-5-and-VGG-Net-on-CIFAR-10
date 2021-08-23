# Steps:
# 1.get the DB from Keras library
# 2.choose a appropriate model
# 3.train the network
# 4.plot the result function
# 5.show the performance of this network

from keras.datasets import cifar10
from keras import layers
from keras import models
from tensorflow.python.keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

# (x_train_original, y_train_original), (x_test_original, y_test_original) = cifar100.load_data(label_mode='fine')

# import time
# start_time = time.time()
# import tensorflow as tf
# a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
# b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
# c = tf.matmul(a, b)
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# print(sess.run(c))
# timer = time.time()
# print(timer - start_time)

def CNN():
    model = models.Sequential()
    model.add(layers.Conv2D(32,(3,3),activation='relu',
                            input_shape=(32, 32,3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


model= CNN()


def getDB():
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    # train_images = train_images.reshape((60000, 32, 32, 3))
    train_images = train_images.astype('float32') / 255
    # test_images = test_images.reshape((10000, 32, 32, 3))
    test_images = test_images.astype('float32') / 255
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    return (train_images, train_labels), (test_images, test_labels)


def fit():
    history = model.fit(train_images, train_labels,
              validation_split= 0.25,
              batch_size=300,
              epochs=10,
              verbose=1)
    return history

def plotResult():
    plt.subplot(211)
    plt.title('Cross-Entropy Loss', pad=-40)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()
    # plot accuracy learning curves
    plt.subplot(212)
    plt.title('Accuracy', pad=-40)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.legend()
    plt.show()

def CNNevaluate():
    train_loss, train_acc = model.evaluate(train_images, train_labels)
    print('\nTrain accuracy: %.4f ' % train_acc)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('\nTest accuracy: %.4f ' % test_acc)
    print('\nPEG= %.4f ' % (1-test_acc))


(train_images, train_labels), (test_images, test_labels) = getDB()
model = CNN()
history = fit()
plotResult()
CNNevaluate()
