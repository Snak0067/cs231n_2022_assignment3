# -*- coding:utf-8 -*-
# @FileName :coco_.py
# @Time :2023/4/18 11:14
# @Author :Xiaofeng
import numpy as np
from matplotlib import pyplot as plt

# Load COCO data from disk into a dictionary.
# We'll work with dimensionality-reduced features for the remainder of this assignment,
# but you can also experiment with the original features on your own by changing the flag below.
from assignment3.cs231n.coco_utils import load_coco_data, sample_coco_minibatch, decode_captions
from assignment3.cs231n.image_utils import image_from_url

data = load_coco_data(pca_features=True)

# Print out all the keys and values from the data dictionary.
for k, v in data.items():
    if type(v) == np.ndarray:
        print(k, type(v), v.shape, v.dtype)
    else:
        print(k, type(v), len(v))

batch_size = 3

captions, features, urls = sample_coco_minibatch(data, batch_size=batch_size)
for i, (caption, url) in enumerate(zip(captions, urls)):
    plt.imshow(image_from_url(url))
    plt.axis('off')
    caption_str = decode_captions(caption, data['idx_to_word'])
    plt.title(caption_str)
    plt.show()
