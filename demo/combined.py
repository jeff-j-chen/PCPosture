import os
import sys
sys.path.append(os.getcwd())
import gc
import cv2
import glob
import torch
import serial


import numpy as np
import depthai as dai
import tensorflow as tf
from IPython import embed
from tensorflow import keras

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patheffects as PathEffects

import libs.model.model as libm
from lib.hrnet.gen_kpts import gen_frame_kpts
from lib.preprocess import h36m_coco_format, revise_kpts
from libs.dataset.h36m.data_utils import unNormalizeData

from model.strided_transformer import Model
from common.camera import normalize_screen_coordinates, camera_to_world
plt.tight_layout()

class PoseAnalyzer:
    def __init__(self, num_joints, pose_connection, re_order_indices, model_path, stats, cascade, input_size, output_size, goodbad_model, figure, threshold, arduino, red, green, font, font_size, font_thickness):
        self.num_joints = num_joints
        self.pose_connection = pose_connection
        self.re_order_indices = re_order_indices
        self.model_path = model_path
        self.stats = stats
        self.cascade = cascade
        self.input_size = input_size
        self.output_size = output_size
        self.goodbad_model = goodbad_model
        self.keypoints = None
        self.figure = figure
        self.threshold = threshold
        self.arduino = arduino
        self.initialize_cascade()
        self.is_available = True
        self.posture_score = threshold
        self.red = red
        self.green = green
        self.font = font
        self.font_size = font_size
        self.font_thickness = font_thickness

    def initialize_cascade(self):
        # initialize cascade?
        for stage_id in range(2):
            stage_model = libm.get_model(
                stage_id + 1,
                refine_3d=False,
                norm_twoD=False,
                num_blocks=2,
                input_size=self.input_size,
                output_size=self.output_size,
                linear_size=1024,
                dropout=0.5,
                leaky=False
            )
            self.cascade.append(stage_model)
        ckpt = torch.load(self.model_path)
        self.cascade.load_state_dict(ckpt)
        self.cascade.eval()

    def gen_pose_2d(self, frame):
        # create 2d keypoints from a given frame
        self.keypoints, scores = gen_frame_kpts(frame, det_dim=416)
        self.keypoints, scores, valid_frames = h36m_coco_format(self.keypoints, scores)
        self.keypoints = revise_kpts(self.keypoints, scores, valid_frames)

    def normalize(self, skeleton, re_order=None):
        # helper function to lift
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

    def get_pred(self, cascade, data):
        # helper function to lift
        num_stages = len(cascade)
        for i in range(len(cascade)):
            cascade[i].num_blocks = len(cascade[i].res_blocks)
        prediction = cascade[0](data)
        for stage_idx in range(1, num_stages):
            prediction += cascade[stage_idx](data)
        return prediction

    def plot_3d_ax(self, ax, elev, azim, pred, title=None):
        # set elev and azim first
        ax.view_init(elev=elev, azim=azim)
        return self.gen_pose_3d(self.re_order(pred), ax)

    def gen_pose_3d(self, channels, ax, gt=False, pred=False):
        # creating matplotlib 3d axes
        # plt.axis('off')
        vals = np.reshape(channels, (32, -1))
        I = np.array([1, 2, 3, 1, 7, 8, 1, 13, 14, 15, 14, 18, 19, 14, 26, 27]) - 1
        J = np.array([2, 3, 4, 7, 8, 9, 13, 14, 15, 16, 18, 19, 20, 26, 27, 28]) - 1
        outs = []
        for i in np.arange(len(I)):
            x, y, z = [((np.array([vals[I[i], j], vals[J[i], j]]) / 500) + 1) / 2 for j in range(3)]
            if (i > 5):
                ax.plot(x, y, z, lw=6)
                ax.scatter(x, y, z, s=150)
                x, y, z = [vals[I[i], j] for j in range(3)]
                outs.append([((x / 500) + 1) / 2, ((y / 500) + 1) / 2, ((z / 500) + 1) / 2])
        ax.set_xlim3d([0, 1])
        ax.set_ylim3d([0, 1])
        ax.set_zlim3d([0, 0.66])
        ax.invert_zaxis()
        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        for line in ax.xaxis.get_ticklines():
            line.set_visible(False)
        for line in ax.yaxis.get_ticklines():
            line.set_visible(False)
        for line in ax.zaxis.get_ticklines():
            line.set_visible(False)
        return outs

    def re_order(self, skeleton):
        # helper function for lifting
        skeleton = skeleton.copy().reshape(-1, 3)
        skeleton[:, [0, 1, 2]] = skeleton[:, [0, 2, 1]]
        skeleton = skeleton.reshape(96)
        return skeleton

    def lift_and_save(self):
        # lifts 2d kpts into 3d kpts and analyzes it
        skeleton_2d = self.keypoints[0][0]
        norm_ske_gt = self.normalize(skeleton_2d, self.re_order_indices).reshape(1, -1)
        pred = self.get_pred(self.cascade, torch.from_numpy(norm_ske_gt.astype(np.float32)))
        pred = unNormalizeData(pred.data.numpy(), self.stats['mean_3d'], self.stats['std_3d'], self.stats['dim_ignore_3d'])
        ax3 = plt.subplot(1, 1, 1, projection='3d')
        outs = self.plot_3d_ax(ax=ax3, pred=pred, elev=10, azim=-105)
        plt.savefig('/home/jeff/Documents/Code/FinalProject/output/pred.png')
        # lift 2d points into 3d and plot it

        proper = np.expand_dims(np.array(outs), axis=0)
        pose_pred = self.goodbad_model.predict(proper)[0]
        final = ""
        if (pose_pred[0] >= pose_pred[1]):
            final = "Good"
            if (self.posture_score < self.threshold):
                self.posture_score += 1
        else:
            final = "Bad"
            if (self.posture_score > -self.threshold):
                self.posture_score -= 1
        if (self.posture_score <= -self.threshold):
            self.arduino.write(str(1).encode())
        else:
            self.arduino.write(str(0).encode())
        # use my model to evaluate good/bad
        to_modify = cv2.imread('/home/jeff/Documents/Code/FinalProject/output/pred.png')
        # put final onto the bottom left corner of the image in bold font
        text_size, base_line = cv2.getTextSize(final, self.font, self.font_size, self.font_thickness)
        text_width,text_height = text_size
        if (final == "Good"):
            font_color = self.green
        elif (final == "Bad"):
            font_color = self.red
        cv2.putText(to_modify, final, (300 - text_width // 2, 550), self.font, self.font_size, font_color, self.font_thickness, cv2.LINE_AA)
        # cv2.imshow("Posture Analyzer", to_modify)
        ax3.clear()

        blank = np.ones((600, 1800, 3), np.uint8) * 255
        blank[0:600, 0:600] = frame[0:600, 150:750]
        blank[0:600, 600:1200] = to_modify
        cv2.line(blank, (600, 0), (600, 600), (0, 0, 0), 4)
        cv2.line(blank, (1200, 0), (1200, 600), (0, 0, 0), 4)
        green_frac = (self.posture_score + self.threshold) / (2 * self.threshold)
        circle_color = (
            round(self.red[0] * (1 - green_frac) + self.green[0] * green_frac),
            round(self.red[1] * (1 - green_frac) + self.green[1] * green_frac),
            round(self.red[2] * (1 - green_frac) + self.green[2] * green_frac),
        )
        text_size, base_line = cv2.getTextSize(str(self.posture_score), self.font, self.font_size, self.font_thickness)
        text_width,text_height = text_size
        cv2.circle(blank, (1500, 300), 200, circle_color, -1, cv2.LINE_AA)
        cv2.circle(blank, (1500, 300), 202, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(blank, str(self.posture_score), (1500 - text_width // 2, 300 + text_height // 2), self.font, self.font_size, (0, 0, 0), self.font_thickness, cv2.LINE_AA)
        # create a black border around the image
        cv2.rectangle(blank, (0, 0), (1800, 600), (0, 0, 0), 8)
        cv2.imshow("Posture Analyzer", blank)

    def fully_process(self, frame):
        '''
        Main function to call, fully processes a given frame.
        '''
        self.is_available = False
        self.gen_pose_2d(frame)
        self.lift_and_save()
        self.is_available = True

    def update_threshold(self, new_threshold):
        self.threshold = new_threshold


cv2.namedWindow("Posture Analyzer", flags=(cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE))
# initial = cv2.imread('/home/jeff/Documents/Code/FinalProject/demo/testimg.png')
# cv2.imshow("Posture Analyzer", initial)

poseAnalyzer = PoseAnalyzer(
    num_joints=16,
    pose_connection=[[0, 1], [1, 2], [2, 3], [3, 4], [2, 5], [5, 6], [6, 7], [2, 8], [8, 9], [9, 10]],
    re_order_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16],
    model_path='/home/jeff/Documents/Code/FinalProject/demo/example_model.th',
    stats=np.load('/home/jeff/Documents/Code/FinalProject/demo/stats.npy', allow_pickle=True).item(),
    cascade=libm.get_cascade(),
    input_size=32,
    output_size=48,
    goodbad_model=tf.keras.models.load_model('/home/jeff/Documents/Code/FinalProject/demo/prettygoodtest.h5'),
    figure=plt.figure(figsize=(6, 6)),
    threshold=5,
    arduino=serial.Serial(port='/dev/ttyACM0', baudrate=9600, timeout=1),
    red=(33*255/100, 35*255/100, 79*255/100),
    green=(33*255/100, 49*255/100, 33*255/100),
    font=cv2.FONT_HERSHEY_DUPLEX,
    font_size=3,
    font_thickness=3,
)


cv2.createTrackbar(
    "Threshold",
    "Posture Analyzer",
    5,
    30,
    poseAnalyzer.update_threshold,
)

gc.collect()
torch.cuda.empty_cache()

pipeline = dai.Pipeline()

camRgb = pipeline.create(dai.node.ColorCamera)
xoutRgb = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")

camRgb.setPreviewSize(900, 600)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

camRgb.preview.link(xoutRgb.input)

with dai.Device(pipeline) as device:

    print('Connected cameras: ', device.getConnectedCameras())
    print('Usb speed: ', device.getUsbSpeed().name)

    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    frameNum = 0
    while True:
        frame = qRgb.get().getCvFrame()
        # cv2.imshow("rgb", frame)
        if poseAnalyzer.is_available:
            poseAnalyzer.fully_process(frame)
        if cv2.waitKey(1) == ord('q'):
            break
