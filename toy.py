import tensorflow as tf

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# print(physical_devices)
# if physical_devices:
#   tf.config.experimental.set_memory_growth(physical_devices[0], True)

# devices = tf.config.list_physical_devices("GPU")
# print(devices)
# print(len(devices))

print(tf.__version__)

# mixed_type = "mixed_float16"
# policy = tf.keras.mixed_precision.Policy(mixed_type)
# print(policy)

# # policy = tf.keras.mixed_precision.experimental.Policy(precision)
# # tf.keras.mixed_precision.experimental.set_policy(policy)

# # MODIFY these lines to
# # policy = tf.keras.mixed_precision.Policy(precision)
# # tf.keras.mixed_precision.set_global_policy(policy)

# import tensorflow_text as text

# # from keras.saving.hdf5_format import save_attributes_to_hdf5_group
# from tensorflow.python.keras.saving.hdf5_format import save_attributes_to_hdf5_group


# print('yes')

import keras 
print(keras.__version__)