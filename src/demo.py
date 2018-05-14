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
from nn import MyNet

parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--model', default='../models/model_best.pth',
                    type=str, help='Trained model path')
parser.add_argument('--live', action='store_true', default=False,
                    help='Switch to live demo mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA')
args = parser.parse_args()

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
POSE_PAIRS = [ ['pelvis','head'],
            ['neck','left_shoulder'], ['left_shoulder','left_elbow'], ['left_elbow','left_hand'],
            ['neck','right_shoulder'], ['right_shoulder','right_elbow'], ['right_elbow','right_hand'],
            ['pelvis','left_hip'], ['left_hip','left_knee'], ['left_knee','left_foot'],
            ['pelvis','right_hip'], ['right_hip','right_knee'], ['right_knee','right_foot'] ]
# all_lines = [[1,0,'b'],[0,2,'b'],[2,3,'b'],[3,4,'b'], #spine
#         [1,5,'b'],[5,6,'b'],[6,7,'b'], #head
#         [5,9,'g'],[9,10,'g'],[10,11,'g'],[11,12,'g'], #left arm
#         [5,14,'r'],[14,15,'r'],[15,16,'r'],[16,17,'r'], #right arm
#         [4,18,'g'],[18,19,'g'],[19,20,'g'],[20,21,'g'], #left lef
#         [4,23,'r'],[23,24,'r'],[24,25,'r'],[25,26,'r']] #right leg
# lines = [[4,6,'b'], # pelvis-head
#         [5,9,'g'],[9,10,'g'],[10,12,'g'], #left arm
#         [5,14,'r'],[14,15,'r'],[15,17,'r'], #right arm
#         [4,18,'g'],[18,19,'g'],[19,21,'g'], #left lef
#         [4,23,'r'],[23,24,'r'],[24,26,'r']] #right leg

def predict(frame, timeit=False):
    x = torch.from_numpy(np.array(frame)).permute(2,0,1).float() / 256
    x = Variable(x.unsqueeze(0))
    if cuda:
        x.cuda()

    start = time.time()
    y = net.forward(x).data.numpy()[0]
    end = time.time()
    if timeit:
        print('the prediction took {:.2f} ms'.format((end-start)*1000.0))

    return y

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
    f.suptitle('Frame, Ground Truth and Prediction')

    ax1 = f.add_subplot(131)
    ax1.imshow(img)

    if len(target):
        ax2 = f.add_subplot(132, projection='3d')
        target = (target+1024)/6.4
        draw_skeleton_plot(ax2, target)

    ax3 = f.add_subplot(133, projection='3d')
    y = predict(img, True)
    draw_skeleton_plot(ax3, y)

    plt.show();

def draw_skeleton_live(frame, y):
    for part in MIN_BODY_PARTS:
        i = BODY_PARTS.index(part)*3
        point = (int(y[i]),int(y[i+1]))
        cv.ellipse(frame, point, (3, 3), 0, 0, 360, (0,0,255), cv.FILLED)
    for pair in POSE_PAIRS:
        idFrom = BODY_PARTS.index(pair[0])*3
        idTo = BODY_PARTS.index(pair[1])*3
        pointFrom = (int(y[idFrom]),int(y[idFrom+1]))
        pointTo = (int(y[idTo]),int(y[idTo+1]))
        cv.line(frame, pointFrom, pointTo, (0, 255, 0), 2)

def demo_live():

    # start video stream thread, allow buffer to fill
    print("[INFO] starting threaded video stream...")
    cap = cv.VideoCapture(0)  # default camera
    num_frames = 0

    # loop over frames from the video file stream
    while True:
        # grab next frame
        _, frame = cap.read()
        key = cv.waitKey(1) & 0xFF
        frame = cv.resize(frame, (320, 320))

        # keybindings for display
        if key == ord('p'):  # pause
            y = predict(frame, True)
            draw_skeleton_live(frame, y)
            while True:
                key2 = cv.waitKey(1) or 0xff
                cv.imshow('My Live Demo', frame)
                if key2 == ord('p'):  # resume
                    break
        cv.imshow('My Live Demo', frame)
        num_frames += 1

        if key == 27:  # exit
            break

    cap.release()
    return frames

if __name__ == '__main__':
    cuda = not args.no_cuda and torch.cuda.is_available()

    net = MyNet()
    net.load_state_dict(torch.load(args.model)['state_dict'])
    net.eval()
    if cuda:
        net.cuda()
    
    if args.live:
        start = time.time()
        num_frames = demo_live()
        elasped = time.time() - start
        print("[INFO] elasped time: {0}".format(elasped))
        print("[INFO] approx. FPS: {:.2f}".format(num_frames / elasped))

        # cleanup
        cv.destroyAllWindows()
    else:
        img = Image.open('../datasets/train/S1_1/S1_1_0_00001.jpg', 'r')        
        target = h5py.File('../datasets/annot.h5', 'r')["S1"]["annot3_1_0"][0]
        create_plot(img, target)

