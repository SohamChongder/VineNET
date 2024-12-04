import cv2
import os
import time
from tqdm import tqdm
import numpy as np
from glob import glob

import tensorflow as tf
from sklearn.metrics import accuracy_score, jaccard_score,f1_score, precision_score, recall_score

import keras
from keras import layers
from keras import ops

import os
import numpy as np
from glob import glob
import cv2
from scipy.io import loadmat
import matplotlib.pyplot as plt

# For data preprocessing
import tensorflow as tf
from tensorflow import image as tf_image
from tensorflow import data as tf_data
from tensorflow import io as tf_io

# IMAGE_SIZE = 512
NUM_CLASSES = 2
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 960

dataset_path = os.path.join("/scratch/ravihm.scee.iitmandi/dataset/vinenet/VineNet-20240427T094946Z-001/VineNet/")
save_path = "/scratch/ravihm.scee.iitmandi/prediction"
model_path = os.path.join("actual_size_image.keras")

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

create_dir(save_path)

model = tf.keras.models.load_model(model_path)

test_x = sorted(glob(os.path.join(dataset_path,"images","*")))
#test_y = sorted(glob(os.path.join(dataset_path,"masks")))

time_log = []
x=0

def read_image(image_path, mask=False):
    image = tf_io.read_file(image_path)
    if mask:
        image = tf_image.decode_png(image, channels=1)
        image.set_shape([None, None, 1])
        image = tf_image.resize(images=image, size=[IMAGE_WIDTH, IMAGE_HEIGHT])
        image = tf.cast(image, tf.float32) / 128.0
    else:
        image = tf_image.decode_png(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf_image.resize(images=image, size=[IMAGE_WIDTH, IMAGE_HEIGHT])
    return image


# inference
colormap = np.array([[0, 0, 0],[255,255,255]])


def infer(model, image_tensor):
    predictions = model.predict(np.expand_dims((image_tensor), axis=0))
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

def plot_samples_matplotlib(display_list, figsize=(5, 3), name = ''):
    cv2.imwrite(os.path.join(save_path,name), display_list[0])
    # _, axes = plt.subplots(nrows=1, ncols=len(display_list), figsize=figsize)
    # for i in range(len(display_list)):
    #     if display_list[i].shape[-1] == 3:
    #         axes.imshow(keras.utils.array_to_img(display_list[i]))
    #     else:
    #         axes.imshow(display_list[i])
    # plt.savefig(os.path.join(save_path,name))
    # plt.close()

def plot_predictions(images_list, colormap, model):
    for image_file in images_list:
        image_tensor = read_image(image_file)
        prediction_mask = infer(image_tensor=image_tensor, model=model)
        prediction_colormap = decode_segmentation_masks(prediction_mask, colormap, 2)
        plot_samples_matplotlib(
            [prediction_colormap], figsize=(20, 10), name = image_file.split('/')[-1]
        )


plot_predictions(test_x, colormap, model=model)


# for x in tqdm(test_x):
#     name = x.split("/")[-1]
#     # print(name)

#     x = read_image(x)
#     # x = cv2.resize(x, (512, 512))
#     # x = x / 255.0
#     # x = np.expand_dims(x, axis=0)
#     # print(x.shape)

#     start_time = time.time()
#     p = infer(model, x)
#     total_time = time.time() - start_time
#     time_log.append(total_time)

#     # print(p.shape)

#     # p = p > 0.5
#     # p = p.astype(np.uint8) * 128  # Convert to uint8 and scale to 255

#     # # Ensure the number of channels is correct
#     # # if len(p.shape) == 2:  # Grayscale image
#     # #     p = cv2.cvtColor(p, cv2.COLOR_GRAY2BGR)
#     # # elif len(p.shape) == 3 and p.shape[2] == 1:  # Single-channel image
#     # #     p = cv2.cvtColor(p, cv2.COLOR_GRAY2BGR)

#     # # print(p.shape)
    

#     cv2.imwrite(os.path.join(save_path, name), p)
#     # x+=1;
#     # if (x==1):
#     #     break



# mean_time = np.mean(np.array(time_log))
# mean_fps = 1 / mean_time

# print(f"Mean_Time: {mean_time: 1.5f} - Mean FPS: {mean_fps:2.5f}")

'''
pred_masks = save_path
#true_masks = os.path.join(write the path here)

score = [[],[],[],[],[]]
for pred_y,true_y in tqdm(zip(pred_masks,true_masks), total = len(pred_masks)):

    pred_y = cv2.imread(pred_y,cv2.IMREAD_GRAYSCALE)
    pred_y = pred_y/128.0
    pred_y = pred_y > 0.5
    pred_y = pred_y.astype(np.int32)
    pred_y = pred_y.flatten()

    true_y = cv2.imread(true_y, cv2.IMREAD_GRAYSCALE)
    true_y = true_y/128.0
    true_y = true_y > 0.5
    true_y = true_y.astype(np.int32)
    true_y = true_y.flatten()

    score[0].append(accuracy_score(pred_y,true_y))
    score[1].append(precision_score(pred_y,true_y,labels = [0,1],average = "binary"))
    score[2].append(recall_score(pred_y,true_y,labels = [0,1],average = "binary"))
    score[3].append(f1_score(pred_y,true_y,labels = [0,1],average = "binary"))
    score[4].append(jaccard_score(pred_y,true_y),labels = [0,1],average = "binary")

mean_score = np.mean(np.array(score), axis = 1)

print(f"Accuracy: {mean_score[0]:0.5f}")
print(f"Recall: {mean_score[3]:0.5f}")
print(f"Precision: {mean_score[4]:0.5f}")
'''