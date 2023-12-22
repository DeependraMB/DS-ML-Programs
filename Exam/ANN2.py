import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.utils import to_categorical,plot_model

(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape(-1,28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

model = Sequential([
    Dense(128, activation='relu', input_shape=(28 * 28,)),
    Dense(68, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=5,batch_size=32,validation_split=0.2)

loss, accuracy = model.evaluate(x_test,y_test)

print(accuracy)

