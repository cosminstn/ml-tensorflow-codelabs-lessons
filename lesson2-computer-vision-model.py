### https://developers.google.com/codelabs/tensorflow-lab2-computervision?continue=https%3A%2F%2Fdevelopers.google.com%2Flearn%2Fpathways%2Ftensorflow%3Fhl%3Den%23codelab-https%3A%2F%2Fdevelopers.google.com%2Fcodelabs%2Ftensorflow-lab2-computervision&hl=en#1

import tensorflow as tf

print(tf.__version__)

mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

import matplotlib.pyplot as plt

plt.imshow(training_images[0])
print(training_labels[0])
print(training_images[0])

training_images = training_images / 255.0
test_images = test_images / 255.0

# Sequential    - defines a sequence of layers in the neural network.
# Flatten       - takes a square and turns it into a one-dimensional vector.
# Dense         - adds a layer of neurons.
# Activation    - functions tell each layer of neurons what to do. There are lots of options, but use these for now:
# Relu          - effectively means that if X is greater than 0 return X, else return 0.
#                 It only passes values of 0 or greater to the next layer in the network.
# Softmax       - takes a set of values, and effectively picks the biggest one.
#                 For example, if the output of the last layer looks like
#                 [0.1, 0.1, 0.05, 0.1, 9.5, 0.1, 0.05, 0.05, 0.05], then it saves you from
#                 having to sort for the largest value—it returns [0,0,0,0,1,0,0,0,0].

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    # tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# Rule of thumb: The number of neurons in the last layer should match the number of classes you are classifying for!
# In this case, it's the digits 0 through 9, so there are 10 of them, and hence you should have 10 neurons
# in your final layer.

model.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# model.fit(training_images, training_labels, epochs=5)

class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.95:
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True


callbacks = MyCallback()
model.fit(training_images, training_labels, epochs=100, callbacks=[callbacks])

model.evaluate(test_images, test_labels)
