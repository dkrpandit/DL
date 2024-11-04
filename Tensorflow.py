import tensorflow as tf
import numpy as np

# Create random input data
x = np.random.rand(100).astype(np.float32)
y = x * 0.2 + 0.2
print(y)

# Initialize Weight and Bias
W = tf.Variable(tf.random.normal([1]))
b = tf.Variable(tf.zeros([1]))

# Create a function for MSE - mean squared error
def mse_loss():
    ypred = W * x + b
    return tf.reduce_mean(tf.square(ypred - y))

# Optimizer
optimizer = tf.keras.optimizers.Adam()

# Iterations
for step in range(10000):
    with tf.GradientTape() as tape:  # Track the gradients
        loss = mse_loss()  # Compute the loss
    gradients = tape.gradient(loss, [W, b])  # Compute gradients of W and b
    optimizer.apply_gradients(zip(gradients, [W, b]))  # Apply gradients to update W and b

    if step % 500 == 0:
        print(step, W.numpy(), b.numpy())
