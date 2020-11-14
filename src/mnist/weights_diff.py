import tensorflow as tf
import numpy as np

model1_dir = 'saved_model/model_0_75_2.h5'
model2_dir = 'saved_model/model_1_75_2.h5'

"""
50:50 split 2 models

mnist/saved_model/model_0_50_2.h5
mnist/saved_model/model_1_50_2.h5

Amount of different weights in Layer 1: 175700 / 235200 (74.70238095238095%)
Maximum Difference in Layer 1: 0.42671263
Amount of different weights in Layer 2: 27089 / 30000 (90.29666666666667%)
Maximum Difference in Layer 2: 0.3940149
Amount of different weights in Layer 3: 9036 / 10000 (90.36%)
Maximum Difference in Layer 3: 0.3818949
Amount of different weights in Layer 4: 17038 / 20000 (85.19%)
Maximum Difference in Layer 4: 0.35703355
Amount of different weights in Layer 5: 1567 / 2000 (78.35%)
Maximum Difference in Layer 5: 0.19004415

75:25 split 2 models

mnist/saved_model/model_0_75_2.h5
mnist/saved_model/model_1_75_2.h5

Amount of different weights in Layer 1: 179038 / 235200 (76.12159863945578%)
Maximum Difference in Layer 1: 0.57485634
Amount of different weights in Layer 2: 27489 / 30000 (91.63%)
Maximum Difference in Layer 2: 0.505146
Amount of different weights in Layer 3: 9114 / 10000 (91.14%)
Maximum Difference in Layer 3: 0.40255263
Amount of different weights in Layer 4: 17359 / 20000 (86.795%)
Maximum Difference in Layer 4: 0.31512654
Amount of different weights in Layer 5: 1618 / 2000 (80.9%)
Maximum Difference in Layer 5: 0.24172486

25:75 split 2 models

mnist/saved_model/model_0_25_2.h5
mnist/saved_model/model_1_25_2.h5

Amount of different weights in Layer 1: 175257 / 235200 (74.5140306122449%)
Maximum Difference in Layer 1: 0.41646057
Amount of different weights in Layer 2: 27091 / 30000 (90.30333333333334%)
Maximum Difference in Layer 2: 0.42387152
Amount of different weights in Layer 3: 8945 / 10000 (89.45%)
Maximum Difference in Layer 3: 0.33507308
Amount of different weights in Layer 4: 16574 / 20000 (82.87%)
Maximum Difference in Layer 4: 0.23994681
Amount of different weights in Layer 5: 1499 / 2000 (74.95%)
Maximum Difference in Layer 5: 0.11338346
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
