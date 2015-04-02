#!/usr/bin/env python
import numpy as np
import pandas as pd
import os
import time
import argparse
import cv2
print cv2.__version__

# Make sure that caffe is on the python path:
caffe_root = '../caffe/'  # this file is expected to be in {caffe_root}/examples

import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

CROP_MODES = ['list', 'selective_search']
COORD_COLS = ['ymin', 'xmin', 'ymax', 'xmax']

def main(argv):
    pycaffe_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    # Required arguments

    # Alternative arguments
    parser.add_argument(
        '--model_def',
        default = os.path.join(pycaffe_dir,
                'R2_hand_classification/lenet.prototxt'),
        help = "Model definition file."
    )
    parser.add_argument(
        '--pretrained_model',
        default = os.path.join(pycaffe_dir,
                'R2_hand_classification/models/train_20X20_iter_10000.caffemodel'),
        help = 'Trianed model weights file.'
    )
    parser.add_argument(
        "--crop_mode",
        default = "list",
        choices = CROP_MODES,
        help = "How to generate windows for detection"
    )
    parser.add_argument(
        '--gpu',
        action = 'store_true',
        help = 'Switch for GPU compution.'
    )
    parser.add_argument(
        '--raw_sacle',
        type = float,
        default = 255.0,
        help = 'Multiply raw input by this scale before preprocessing.'
    )
    parser.add_argument(
        '--context_pad',
        type = int,
        default = '16',
        help = "Amount of surrounding context to collect in input window."
    )
    args = parser.parse_args()

    channel_swap = None

    # Make Detector
    detector = caffe.Detector(args.model_def, args.pretrained_model,
            gpu = args.gpu, raw_scale = args.raw_scale,
            channel_swap = channel_swap,
            context_pad = args.context_pad)

    # Load input
    t = time.time()

    # Detect
    if args.crop_mode == 'list':


# outpath = ('output.avi')
# fps = 20.0
#
# cap = cv2.VideoCapture(0)
# if not cap:
#     print "!!! Failed VideoCapture: invalid parameter"
#     sys.exit(1)
#
# # The following might fail if the device doesn't these values
# cap.set(1, fps) # Match fps
# cap.set(3, 640)  # Match width
# cap.set(4, 480) # Match height
#
# # So it's always safer to trtrieve it afterwards
# w = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
# h = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
# print type(fps)
# print type(w)
# print type(h)
# print fps
#
# # Define the codec and create VideoWriter boject
# fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
#
# video_writer = cv2.VideoWriter(outpath, fourcc, fps, (int(w), int(h)), True)
#
# if not video_writer:
#     print "!!! Failed VideoWriter: invalid parameters"
#     sys.exit(1)
#
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret == False:
#         break
#     cv2.imshow('frame', frame)
#     video_writer.write(frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release everything if job is finished
# cap.release()
# video_writer.release()
# cv2.destroyAllWindows()

class CaptureManager(object):
    def __init__(self, capture, previewWindowManager = None,
                shouldMirrorPreview = False):
        self. previewWindowManager = previewWindowManger
        self. shouldMirrorPreview = shouldMirrorPreview

        self._capture = capture
        self._channel = 0
        self._enteredFrame = False
        self._frame = None
        self._imageFilename = None
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter = None

        self._startTime = None
        self._frameElapsed = long(0)
        self._fpsEstimate = None

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, value):
        if self._channel != value:
            self._channel = value
            self._frame = None

    @property
    def frame(self):
        if self._enteredFrame and self._frame is None:
            _, self._frame = self._capture.retrieve(
                channel = self.channel)
        return self._frame

    @property
    def isWritingImage(self):
        return self._imageFilename is not None

    @property
    def isWritingVideo(self):
        return self._videoFilename is not None

    def enterFrame(self):
        """Capture the next frame, if any."""

        # But frist, check that any previous frame was exited
        assert not self._enteredFrame, \
            'previous enterFrame() had not matching exitFrame()'

        if self._capture is not None:
            self._enteredFrame = self._capture.grab()

    def exitFrame (self):
        """Draw to the window. Write to files. Release the frame"""

if __name__ == "__main__":
    import sys
    main(sys.argv)
