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


def get_pose2D(frames, output_dir):
    print('\nGenerating 2D pose...')
    keypoints, scores = gen_frame_kpts(frames, det_dim=416, num_people=1, gen_output=True)
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
    keypoints = revise_kpts(keypoints, scores, valid_frames)
    # save keypoints as a .npy file
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'keypoints.npy'), keypoints)
    print('Generating 2D pose successful!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='sample_video.mp4', help='input video')
    args = parser.parse_args()

    output_dir = '/home/jeff/Documents/Code/Final_Project/output/'
    frame = cv2.imread('/home/jeff/Documents/Code/Final_Project/demo/testimg.png')
    cv2.imshow('frame', frame)
    get_pose2D(frame, output_dir)
    print("done")