import sys
import argparse
import cv2
from lib.preprocess import h36m_coco_format, revise_kpts
from lib.hrnet.gen_kpts import gen_video_kpts, gen_frame_kpts
import os
import numpy as np
import torch
import glob
from tqdm import tqdm
import copy
from IPython import embed

sys.path.append(os.getcwd())
from model.strided_transformer import Model
from common.camera import normalize_screen_coordinates, camera_to_world

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
tf.keras.backend.clear_session()
from tensorflow.keras.layers import Input, Softmax, Dense, Dropout, BatchNormalization
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras.applications import VGG16
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import random

import gc


gc.collect()

torch.cuda.empty_cache()

keypoints = None
def get_pose2D(frames):
    global keypoints
    print('\nGenerating 2D pose...')
    keypoints, scores = gen_frame_kpts(frames, det_dim=416)
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
    keypoints = revise_kpts(keypoints, scores, valid_frames)
    # save keypoints as a .npy file
    print('Generating 2D pose successful!')


frame = cv2.imread('/home/jeff/Documents/Code/FinalProject/demo/testimg.png')
get_pose2D(frame)



import sys
# sys.path.append("../")

import libs.model.model as libm
from libs.dataset.h36m.data_utils import unNormalizeData
import cv2
import torch
import numpy as np
import imageio
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

num_joints = 16
gt_3d = False
pose_connection = [[0, 7 - 6], [7 - 6, 8 - 6], [8 - 6, 9 - 6], [9 - 6, 10 - 6], [8 - 6, 11 - 6], [11 - 6, 12 - 6], [12 - 6, 13 - 6], [8 - 6, 14 - 6], [14 - 6, 15 - 6], [15 - 6, 16 - 6]]
re_order_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16]
# paths
model_path = '/home/jeff/Documents/Code/FinalProject/demo/example_model.th'
stats = np.load('/home/jeff/Documents/Code/FinalProject/demo/stats.npy', allow_pickle=True).item()
dim_used_2d = stats['dim_use_2d']
mean_2d = stats['mean_2d']
std_2d = stats['std_2d']
ckpt = torch.load(model_path)
cascade = libm.get_cascade()
input_size = 32
output_size = 48
for stage_id in range(2):
    stage_model = libm.get_model(stage_id + 1,
                                 refine_3d=False,
                                 norm_twoD=False,
                                 num_blocks=2,
                                 input_size=input_size,
                                 output_size=output_size,
                                 linear_size=1024,
                                 dropout=0.5,
                                 leaky=False)
    cascade.append(stage_model)
cascade.load_state_dict(ckpt)
cascade.eval()
count = 0
total_to_show = 10


def normalize(skeleton, re_order=None):
    norm_skel = skeleton.copy()
    if re_order is not None:
        norm_skel = norm_skel[re_order].reshape(32)
    norm_skel = norm_skel.reshape(16, 2)
    mean_x = np.mean(norm_skel[:, 0])
    std_x = np.std(norm_skel[:, 0])
    mean_y = np.mean(norm_skel[:, 1])
    std_y = np.std(norm_skel[:, 1])
    denominator = (0.5 * (std_x + std_y))
    norm_skel[:, 0] = (norm_skel[:, 0] - mean_x) / denominator
    norm_skel[:, 1] = (norm_skel[:, 1] - mean_y) / denominator
    norm_skel = norm_skel.reshape(32)
    return norm_skel


def get_pred(cascade, data):
    num_stages = len(cascade)
    for i in range(len(cascade)):
        cascade[i].num_blocks = len(cascade[i].res_blocks)
    prediction = cascade[0](data)
    for stage_idx in range(1, num_stages):
        prediction += cascade[stage_idx](data)
    return prediction


def show3Dpose(channels, ax, gt=False, pred=False):
    vals = np.reshape(channels, (32, -1))
    I = np.array([1, 2, 3, 1, 7, 8, 1, 13, 14, 15, 14, 18, 19, 14, 26, 27]) - 1
    J = np.array([2, 3, 4, 7, 8, 9, 13, 14, 15, 16, 18, 19, 20, 26, 27, 28]) - 1
    outs = []
    for i in np.arange(len(I)):
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        x = ((x / 500) + 1) / 2
        y = ((y / 500) + 1) / 2
        z = ((z / 500) + 1) / 2
        if (i > 5):
            ax.plot(x, y, z, lw=7)
            ax.scatter(x, y, z, s=200)
            x, y, z = [vals[I[i], j] for j in range(3)]
            outs.append([((x / 500) + 1) / 2, ((y / 500) + 1) / 2, ((z / 500) + 1) / 2])
    ax.set_xlim3d([0, 1])
    ax.set_ylim3d([0, 1])
    ax.set_zlim3d([0, 0.75])
    white = (1.0, 1.0, 1.0, 0.0)
    ax.w_xaxis.set_pane_color(white)
    ax.w_yaxis.set_pane_color(white)
    ax.w_xaxis.line.set_color(white)
    ax.w_yaxis.line.set_color(white)
    ax.w_zaxis.line.set_color(white)
    ax.invert_zaxis()
    return outs


def re_order(skeleton):
    skeleton = skeleton.copy().reshape(-1, 3)
    skeleton[:, [0, 1, 2]] = skeleton[:, [0, 2, 1]]
    skeleton = skeleton.reshape(96)

    return skeleton


def plot_3d_ax(ax, elev, azim, pred, title=None):
    ax.view_init(elev=elev, azim=azim)
    return show3Dpose(re_order(pred), ax)

test_model = tf.keras.models.load_model('/home/jeff/Documents/Code/FinalProject/demo/prettygoodtest.h5')
val_model = tf.keras.models.load_model('/home/jeff/Documents/Code/FinalProject/demo/prettygoodval.h5')
f = plt.figure(figsize=(15, 15))

t = np.load("/home/jeff/Documents/Code/FinalProject/output/keypoints.npy")

skeleton_2d = keypoints[0][0]
norm_ske_gt = normalize(skeleton_2d, re_order_indices).reshape(1, -1)
pred = get_pred(cascade, torch.from_numpy(norm_ske_gt.astype(np.float32)))
pred = unNormalizeData(pred.data.numpy(), stats['mean_3d'], stats['std_3d'], stats['dim_ignore_3d'] )
ax3 = plt.subplot(1, 1, 1, projection='3d')
outs = plot_3d_ax(ax=ax3, pred=pred, elev=5, azim=-80)
plt.savefig('/home/jeff/Documents/Code/FinalProject/output/pred.png')
img = cv2.imread('/home/jeff/Documents/Code/FinalProject/output/pred.png')
proper = np.expand_dims(np.array(outs), axis=0)
pose_pred = val_model.predict(proper)[0]
final = ""
if (pose_pred[0] >= pose_pred[1]):
    final = "good"
else:
    final = "bad"
print(final)
cv2.imwrite('/home/jeff/Documents/Code/FinalProject/output/pred.png', img)
ax3.clear()
