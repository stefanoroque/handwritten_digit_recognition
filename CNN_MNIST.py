# Load in Keras
import keras

# Load in the MNIST data set
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()


# Must explicitly declare that our images have a depth of 1 (they are grey-scale)
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)
# Change data type to float so that we can rescale
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
# Rescale the data so that each pixel value is between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0
# Sort the y data into 10 different bins (there are 10 different digits/categories)
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Initialize a sequential model
convolutional_nn = keras.models.Sequential()
# Add first convolutional layer
convolutional_nn.add(keras.layers.Conv2D(32, kernel_size=(5,5), input_shape=(28,28,1), padding="same", activation="relu"))
# Add first max pooling layer
convolutional_nn.add(keras.layers.MaxPooling2D())
# Add second convolutional layer
# Keras can infer the input shape so we don't need to specify
convolutional_nn.add(keras.layers.Conv2D(64, kernel_size=(5,5), padding="same", activation="relu"))
# Add second max pooling layer
convolutional_nn.add(keras.layers.MaxPooling2D())
# Flatten the neural net because we have a fully connected layer coming next
convolutional_nn.add(keras.layers.Flatten())
# Add dropout to help prevent over fitting
convolutional_nn.add(keras.layers.Dropout(0.4))
# Add our fully connected/dense layer
# Uses 3136 nodes because the input before we flatten it is 64 filters with dimensions of 7x7 (64*7*7 = 3136)
convolutional_nn.add(keras.layers.Dense(3136, activation="relu"))
# Add second fully connected layer (the output layer)
convolutional_nn.add(keras.layers.Dense(10, activation="softmax"))

convolutional_nn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Print a summary of the model so that we can see how it is set up
print(convolutional_nn.summary())


# Train the model (this will take a while)
training_history = convolutional_nn.fit(X_train, y_train, epochs=10, validation_data=(X_test,y_test), verbose=2)

# Evaluate the performance of the model
performance = convolutional_nn.evaluate(X_test, y_test)
print("")
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("The accuracy on the testing data after training the model is {}%".format(performance[1]*100))
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


