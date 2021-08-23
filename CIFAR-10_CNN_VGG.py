# 导包
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np

(x_train,y_train),(x_test,y_test) = cifar10.load_data()
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

model = Sequential()

# block1
model.add(Conv2D(filters=64,kernel_size=(3,3),activation="relu",padding="same",name="block1_conv1",input_shape=(32,32,3)))
model.add(Conv2D(filters=64,kernel_size=(3,3),activation="relu",padding="same",name="block1_conv2"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),name="block1_pool"))
# block2
model.add(Conv2D(filters=128,kernel_size=(3,3),activation="relu",padding="same",name="block2_conv1"))
model.add(Conv2D(filters=128,kernel_size=(3,3),activation="relu",padding="same",name="block2_conv2"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),name="block2_pool"))
# block3
model.add(Conv2D(filters=256,kernel_size=(3,3),activation="relu",padding="same",name="block3_conv1"))
model.add(Conv2D(filters=256,kernel_size=(3,3),activation="relu",padding="same",name="block3_conv2"))
model.add(Conv2D(filters=256,kernel_size=(3,3),activation="relu",padding="same",name="block3_conv3"))
model.add(Conv2D(filters=256,kernel_size=(3,3),activation="relu",padding="same",name="block3_conv4"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),name="block3_pool"))
# block4
model.add(Conv2D(filters=512,kernel_size=(3,3),activation="relu",padding="same",name="block4_conv1"))
model.add(Conv2D(filters=512,kernel_size=(3,3),activation="relu",padding="same",name="block4_conv2"))
model.add(Conv2D(filters=512,kernel_size=(3,3),activation="relu",padding="same",name="block4_conv3"))
model.add(Conv2D(filters=512,kernel_size=(3,3),activation="relu",padding="same",name="block4_conv4"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),name="block4_pool"))
# block5
model.add(Conv2D(filters=512,kernel_size=(3,3),activation="relu",padding="same",name="block5_conv1"))
model.add(Conv2D(filters=512,kernel_size=(3,3),activation="relu",padding="same",name="block5_conv2"))
model.add(Conv2D(filters=512,kernel_size=(3,3),activation="relu",padding="same",name="block5_conv3"))
model.add(Conv2D(filters=512,kernel_size=(3,3),activation="relu",padding="same",name="block5_conv4"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),name="block5_pool"))

model.add(Flatten())
model.add(Dense(4096,activation="relu",name="fc1"))
# model.add(Dropout(0.5))
model.add(Dense(4096,activation="relu",name="fc2"))
# model.add(Dropout(0.5))
model.add(Dense(10,activation="softmax",name="prediction"))
model.summary()

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# compile
model.compile(loss="categorical_crossentropy",optimizer=sgd,metrics=['accuracy'])
# train
history=model.fit(x_train,y_train,epochs=50,batch_size=64,validation_split=0.1,verbose=1)

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

train_loss, train_acc = model.evaluate(x_train, y_train)
print('\nTrain accuracy: %.4f ' % train_acc)
test_loss, test_acc = model.evaluate(x_test, y_test)
print('\nTest accuracy: %.4f ' % test_acc)
print('\nPEG= %.4f ' % (1 - test_acc))