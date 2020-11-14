import tensorflow as tf
import numpy as np

"""
50:50 split among 2 models

mnist-fashion/models/clip_1_5_NO_0_50_2.h5
mnist-fashion/models/clip_1_5_NO_1_50_2.h5

Amount of different weights in Layer 1: 92142 / 1176000 (7.835204081632654%)
Maximum Difference in Layer 1: 0.10004058
Amount of different weights in Layer 2: 93391 / 1500000 (6.226066666666666%)
Maximum Difference in Layer 2: 0.11261404
Amount of different weights in Layer 3: 45664 / 500000 (9.132800000000001%)
Maximum Difference in Layer 3: 0.104167596
Amount of different weights in Layer 4: 715 / 5000 (14.299999999999999%)
Maximum Difference in Layer 4: 0.08334696
Amount of different weights in Layer 5: 522 / 5000 (10.440000000000001%)
Maximum Difference in Layer 5: 0.07210133
Amount of different weights in Layer 6: 75109 / 500000 (15.021799999999999%)
Maximum Difference in Layer 6: 0.22603089
Amount of different weights in Layer 7: 220529 / 1500000 (14.701933333333333%)
Maximum Difference in Layer 7: 0.3120862
Amount of different weights in Layer 8: 265609 / 1176000 (22.585799319727894%)
Maximum Difference in Layer 8: 0.29135352

25:50 split among 2 models

mnist-fashion/models/clip_1_5_NO_0_25_2.h5
mnist-fashion/models/clip_1_5_NO_1_25_2.h5

Amount of different weights in Layer 1: 84267 / 1176000 (7.165561224489796%)
Maximum Difference in Layer 1: 0.07408059
Amount of different weights in Layer 2: 84613 / 1500000 (5.640866666666667%)
Maximum Difference in Layer 2: 0.11220543
Amount of different weights in Layer 3: 43264 / 500000 (8.6528%)
Maximum Difference in Layer 3: 0.094080076
Amount of different weights in Layer 4: 816 / 5000 (16.32%)
Maximum Difference in Layer 4: 0.036029667
Amount of different weights in Layer 5: 730 / 5000 (14.6%)
Maximum Difference in Layer 5: 0.06680465
Amount of different weights in Layer 6: 74859 / 500000 (14.971799999999998%)
Maximum Difference in Layer 6: 0.16774371
Amount of different weights in Layer 7: 207337 / 1500000 (13.822466666666667%)
Maximum Difference in Layer 7: 0.22526443
Amount of different weights in Layer 8: 231951 / 1176000 (19.72372448979592%)
Maximum Difference in Layer 8: 0.25852636

75:25 split among 2 models

mnist-fashion/models/clip_1_5_NO_0_75_2.h5
mnist-fashion/models/clip_1_5_NO_1_75_2.h5

Amount of different weights in Layer 1: 101609 / 1176000 (8.640221088435373%)
Maximum Difference in Layer 1: 0.07751802
Amount of different weights in Layer 2: 103581 / 1500000 (6.9054%)
Maximum Difference in Layer 2: 0.13161524
Amount of different weights in Layer 3: 44504 / 500000 (8.9008%)
Maximum Difference in Layer 3: 0.14839369
Amount of different weights in Layer 4: 836 / 5000 (16.72%)
Maximum Difference in Layer 4: 0.020743955
Amount of different weights in Layer 5: 662 / 5000 (13.239999999999998%)
Maximum Difference in Layer 5: 0.062371463
Amount of different weights in Layer 6: 89151 / 500000 (17.830199999999998%)
Maximum Difference in Layer 6: 0.16287407
Amount of different weights in Layer 7: 261588 / 1500000 (17.4392%)
Maximum Difference in Layer 7: 0.283146
Amount of different weights in Layer 8: 299188 / 1176000 (25.441156462585035%)
Maximum Difference in Layer 8: 0.3264294
"""

model1_dir = 'models/clip_1_5_NO_0_75_2.h5'
model2_dir = 'models/clip_1_5_NO_1_75_2.h5'

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
