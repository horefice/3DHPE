# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import cv2 as cv
import argparse
import time
import h5py
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from mpl_toolkits.mplot3d import Axes3D
from torch.autograd import Variable
from nn import MyNet, VNect

parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--model', type=str, default='../models/model_best.pth',
                    help='Trained model path')
parser.add_argument('--live', action='store_true', default=False,
                    help='Switch to live demo mode')
parser.add_argument('--video', type=str, default='',
                    help='Use video as input')
parser.add_argument('--image', type=str, default='',
                    help='Use image as input')
parser.add_argument('--vnect', action='store_true', default=False,
                    help='uses VNect-like network')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# DATASET INFO
#   all_joint_names = {'spine3', 'spine4', 'spine2', 'spine', 'pelvis', ...     %5       
#         'neck', 'head', 'head_top', 'left_clavicle', 'left_shoulder', 'left_elbow', ... %11
#        'left_wrist', 'left_hand',  'right_clavicle', 'right_shoulder', 'right_elbow', 'right_wrist', ... %17
#        'right_hand', 'left_hip', 'left_knee', 'left_ankle', 'left_foot', 'left_toe', ...        %23   
#        'right_hip' , 'right_knee', 'right_ankle', 'right_foot', 'right_toe'}; 
#         joint_parents_o1 = [3, 1, 4, 5, 5, 2, 6, 7, 6, 9, 10, 11, 12, 6, 14, 15, 16, 17, 5, 19, 20, 21, 22, 5, 24, 25, 26, 27 ];
#         joint_parents_o2 = [4, 3, 5, 5, 5, 1, 2, 6, 2, 6, 9, 10, 11, 2, 6, 14, 15, 16, 4, 5, 19, 20, 21, 4, 5, 24, 25, 26];

# GRAPHS
BODY_PARTS = ['spine3', 'spine4', 'spine2', 'spine', 'pelvis',  
              'neck', 'head', 'head_top', 'left_clavicle', 'left_shoulder', 'left_elbow',
              'left_wrist', 'left_hand',  'right_clavicle', 'right_shoulder', 'right_elbow', 'right_wrist',
              'right_hand', 'left_hip', 'left_knee', 'left_ankle', 'left_foot', 'left_toe',
              'right_hip' , 'right_knee', 'right_ankle', 'right_foot', 'right_toe']
MIN_BODY_PARTS = ['pelvis','head','left_hand','right_hand','left_foot','right_foot']
POSE_PAIRS = [['pelvis','head'],
              ['neck','left_shoulder'], ['left_shoulder','left_elbow'], ['left_elbow','left_hand'],
              ['neck','right_shoulder'], ['right_shoulder','right_elbow'], ['right_elbow','right_hand'],
              ['pelvis','left_hip'], ['left_hip','left_knee'], ['left_knee','left_foot'],
              ['pelvis','right_hip'], ['right_hip','right_knee'], ['right_knee','right_foot']]

def predict(frame, timeit=False):
  net.eval()

  x = torch.from_numpy(np.array(frame)).permute(2,0,1).float() / 256
  x = Variable(x.unsqueeze(0))
  if net.is_cuda:
    x = x.cuda()

  start = time.time()
  y = net.forward(x)
  if timeit:
    print('Prediction function took {:.2f} ms'.format((time.time()-start)*1000.0))
  y = y.data.cpu().numpy()[0]

  return y

def rescale(x):
  return int((x+1024)*320/2048)

def draw_skeleton_plot(plot, target):
  for pose in POSE_PAIRS:
    i1 = BODY_PARTS.index(pose[0])*3
    i2 = BODY_PARTS.index(pose[1])*3
    plot.plot([target[i1],target[i2]],
        [target[i1+1],target[i2+1]],
        [target[i1+2],target[i2+2]], 'g')
  for part in MIN_BODY_PARTS:
    i = BODY_PARTS.index(part)*3
    plot.plot([target[i]],[target[i+1]],[target[i+2]], 'o')

  plot.set_xlabel('X')
  plot.set_ylabel('Y')
  plot.set_zlabel('Z')
  plot.view_init(-90,-90)

def create_plot(img, target=[]):
  f = plt.figure()
  f.suptitle('Demo from Image')

  ax1 = f.add_subplot(131)
  ax1.set_title('Input')
  ax1.imshow(img)

  if len(target):
    ax2 = f.add_subplot(132, projection='3d')
    ax2.set_title('Ground Truth')
    target = target
    draw_skeleton_plot(ax2, target)

  ax3 = f.add_subplot(133, projection='3d')
  ax3.set_title('Prediction')
  y = predict(img, True)
  draw_skeleton_plot(ax3, y)

  plt.show();

def draw_skeleton_live(frame, y):
  for part in MIN_BODY_PARTS:
    i = BODY_PARTS.index(part)*3
    point = (rescale(y[i]),rescale(y[i+1]))
    cv.ellipse(frame, point, (3, 3), 0, 0, 360, (0,0,255), cv.FILLED)
  for pair in POSE_PAIRS:
    idFrom = BODY_PARTS.index(pair[0])*3
    idTo = BODY_PARTS.index(pair[1])*3
    pointFrom = (rescale(y[idFrom]),rescale(y[idFrom+1]))
    pointTo = (rescale(y[idTo]),rescale(y[idTo+1]))
    cv.line(frame, pointFrom, pointTo, (0, 255, 0), 1)

def demo_video(video_path=0):

  # start video stream thread, allow buffer to fill
  print("[INFO] starting threaded video stream...")
  cap = cv.VideoCapture(video_path) # 0 for default camera
  num_frames = 0
  time_paused = 0
  start = time.time()

  # loop over frames from the video file stream
  while cap.isOpened():
    # grab next frame
    isFrame, frame = cap.read()
    if not isFrame:
      break
    key = cv.waitKey(1) & 0xFF
    frame = cv.resize(frame, (320, 320))

    if args.cuda:
      y = predict(frame)
      draw_skeleton_live(frame, y)
    
    # keybindings for display
    if key == ord('p'):  # pause
      start_pause = time.time()
      if not args.cuda:
        y = predict(frame)
        draw_skeleton_live(frame, y)

      while True:
        key2 = cv.waitKey(1) or 0xff
        cv.imshow('Video Demo', frame)
        if key2 == ord('p'):  # resume
          time_paused += time.time() - start_pause
          break

    cv.imshow('Video Demo', frame)
    num_frames += 1

    if key == 27:  # exit
      break

  elasped = time.time() - start
  print("[INFO] elasped time: {:.2f}s".format(elasped))
  print("[INFO] approx. FPS: {:.2f}".format(num_frames / (elasped-time_paused)))
  
  cap.release()
  cv.destroyAllWindows()

if __name__ == '__main__':
  net = MyNet() if not args.vnect else VNect()
  net.load_state_dict(torch.load(args.model)['state_dict'])
  if args.cuda:
    net.cuda()
  
  if args.live:
    demo_video(0)
  elif args.video:
    demo_video(args.video)
  elif args.image:
    img = Image.open(args.image, 'r')
    img = img.resize((320, 320), Image.ANTIALIAS)
    create_plot(img)
  else:
    img = Image.open('../datasets/train/S1_1/S1_1_0_00001.jpg', 'r')        
    target = h5py.File('../datasets/annot.h5', 'r')["S1"]["annot3_1_0"][0]
    create_plot(img, target)
