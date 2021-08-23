# Cross Validation Classification Confusion Matrix
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from keras.layers import Dropout

print('\nDNN_CIFAR10')




def train( numOfEpochs, bantchSize):
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    train_images = train_images.reshape((50000, 32 * 32 * 3))
    train_images = train_images.astype('float32') / 255
    test_images = test_images.reshape((10000, 32 * 32 * 3))
    test_images = test_images.astype('float32') / 255

    network = Sequential()
    network.add(Dense(32, activation='relu'))
    network.add(Dense(64, activation='relu'))
    network.add(Dense(128, activation='relu'))
    network.add(Dense(10, activation='softmax'))
    network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    history = network.fit(train_images, train_labels, epochs=numOfEpochs, batch_size=bantchSize, validation_split=0.2)
    network.summary()
    test_loss, test_acc = network.evaluate(test_images, test_labels)
    print("The accuracy of test sets is: ")
    print(test_acc)
    return test_acc, history


test_acc, history = train(100, 200)

plt.subplot(2,1,1)
plt.title('Cross-Entropy Loss', pad=-40)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
# plot accuracy learning curves

plt.subplot(2,1,2)
plt.title('Accuracy', pad=-40)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.legend()
plt.show()
