import tensorflow as tf
import numpy as np

model1_dir = 'models/clip_1_5_NO_0_50_2.h5'
model2_dir = 'models/clip_1_5_NO_1_50_2.h5'

"""
50:50 split among 2 models:

storage_testing/models/clip_1_5_NO_0_50_2.h5
storage_testing/models/clip_1_5_NO_1_50_2.h5

Amount of different weights in Layer 1: 186 / 1024 (18.1640625%)
Maximum Difference in Layer 1: 0.025714226
Amount of different weights in Layer 2: 129 / 8192 (1.57470703125%)
Maximum Difference in Layer 2: 0.022770114
Amount of different weights in Layer 3: 115 / 16384 (0.701904296875%)
Maximum Difference in Layer 3: 0.022501048
Amount of different weights in Layer 4: 49 / 320 (15.312500000000002%)
Maximum Difference in Layer 4: 0.023929954
"""

model1 = tf.keras.models.load_model(model1_dir)
model2 = tf.keras.models.load_model(model2_dir)

model_diff = []
layer_sizes = []
for i in range(len(model1.layers)):
  if len(model1.layers[i].get_weights()) > 0:
    print("Layer " + str(i + 1) + ":")
    layer_diff = model1.layers[i].get_weights()[0] - model2.layers[i].get_weights()[0]
    model_diff.append(layer_diff)
    print(layer_diff)
for i in range(len(model_diff)):
  current_layer_size = 0
  total_nonzero = 0
  max = 0
  for cell in np.nditer(model_diff[i]):
    current_layer_size += 1
    if abs(cell) > 0.01:
      total_nonzero += 1
      if abs(cell) > max:
        max = cell
  percentage_diff = ((total_nonzero * 1.) / current_layer_size) * 100
  print("Amount of different weights in Layer " + str(i + 1) + ": " + str(total_nonzero)
        + " / " + str(current_layer_size) + " (" + str(percentage_diff) + "%)")
  print("Maximum Difference in Layer " + str(i+1) + ": " + str(max))
  layer_sizes.append(current_layer_size)

#for size in layer_sizes:
  #print(size)
