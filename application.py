from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow import image as tf_image
from tensorflow import io as tf_io
import matplotlib.pyplot as plt
from PIL import Image as im


# Load the pre-trained model and define necessary variables
model_path = "actual_size_image.keras"
model = tf.keras.models.load_model(model_path)

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 960
colormap = np.array([[0, 0, 0], [255, 255, 255]])

def process_image():
    print("Entered")
    image_data=tf_io.read_file("./static/input.png")
    segmented_image = segment_image(image_data)
    data = im.fromarray(segmented_image)
    data.save('./static/output.png')
    print("Exited")

# image_data = tf_io.read_file("/Users/sohamchongder/Downloads/Block_5E1_Row_1_Middle_4086.png")


def segment_image(image_data):
    image = tf_image.decode_image(image_data)
    # image = tf_image.resize(image, [IMAGE_WIDTH, IMAGE_HEIGHT])
    prediction_mask = infer(model, image)
    prediction_colormap = decode_segmentation_masks(prediction_mask, colormap, 2)
    return prediction_colormap

def infer(model, image_tensor):
    predictions = model.predict(np.expand_dims(image_tensor, axis=0))
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=2)
    return predictions

def decode_segmentation_masks(mask, colormap, n_classes):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(0, n_classes):
        idx = mask == l
        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb

# x = segment_image(image_data)
# print(type(x))
# plt.imshow(x)
# plt.show()