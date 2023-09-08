import time
import tensorflow as tf

print("TensorFlow version:", tf.__version__)

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))

# Create two random matrices
matrix_size = 1000
matrix_a = tf.random.normal((matrix_size, matrix_size))
matrix_b = tf.random.normal((matrix_size, matrix_size))

# Perform matrix multiplication
start_time = time.time()
result = tf.matmul(matrix_a, matrix_b)
end_time = time.time()

print(f"Matrix multiplication took {end_time - start_time:.4f} seconds")

# Verify the result
if tf.reduce_sum(result).numpy() < 0:
    print("Result is negative.")
else:
    print("Result is non-negative.")