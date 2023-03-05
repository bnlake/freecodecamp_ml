import tensorflow as tf

# Rank/Degree 0 tensor (also known as a scalar tensor) (Not contained within an array)
string = tf.Variable("This is a string", tf.string)
number = tf.Variable(12, tf.int16)
floating = tf.Variable(3.14, tf.float16)
print(tf.rank(number))

# Rank/Degree 1 tensor
tensor_rank1 = tf.Variable([1, 2, 4], tf.int16)
print(tf.rank(tensor_rank1))

# Rank/Degree 2 tensor (aka Matrice)
tensor_rank2 = tf.Variable([[1, 2], [3, 4]], tf.int16)
print(tf.rank(tensor_rank2))


t = tf.ones([2, 1, 3])

t = tf.reshape(t, [2, -1])
print(t)