#!/usr/bin/env python3

import cv2
import depthai as dai

pipeline = dai.Pipeline()

camRgb = pipeline.create(dai.node.ColorCamera)
xoutRgb = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")

camRgb.setPreviewSize(600, 600)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

camRgb.preview.link(xoutRgb.input)

with dai.Device(pipeline) as device:

    print('Connected cameras: ', device.getConnectedCameras())
    print('Usb speed: ', device.getUsbSpeed().name)

    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    frameNum = 0
    while True:
        inRgb = qRgb.get()
        frameNum += 1
        if (frameNum % 60 == 0):
            cv2.imshow("rgb", inRgb.getCvFrame())
        if cv2.waitKey(1) == ord('q'):
            break
