import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical,plot_model
import matplotlib.pyplot as plt

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0

# Flatten the images (convert 28x28 images to 1D vectors)
X_train = X_train.reshape(-1, 28 * 28)
X_test = X_test.reshape(-1, 28 * 28)

# One-hot encode the target labels
y_train = to_categorical(y_train, 10)  # 10 is the number of classes
y_test = to_categorical(y_test, 10)

# plt.figure(figsize=(10, 5))
# for i in range(100):  # Display 10 samples
#     plt.subplot(20, 50, i + 1)
#     plt.imshow(X_train[i].reshape(28, 28), cmap='gray')
#     plt.title(f"Label: {y_train[i].argmax()}")  # Display the one-hot encoded label
#
# plt.show()

# Create a simple feedforward neural network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(28 * 28,)),
    Dense(68, activation='relu'),
    Dense(10, activation='softmax')
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")
