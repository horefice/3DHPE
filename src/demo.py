import numpy as np
import torch
import cv2
import argparse
import time
from imutils.video import FPS
from torch.autograd import Variable
from nn import MyNet

parser = argparse.ArgumentParser(description='Live Demo')
parser.add_argument('--model', default='../models/nn.pth',
                    type=str, help='Trained model path')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA')
args = parser.parse_args()

def cv2_demo(net):
    def predict(frame):
        frame = cv2.resize(frame, (320, 320))

        x = torch.from_numpy(frame.astype(np.float32)).permute(2,0,1)
        x = Variable(x.unsqueeze(0))
        if cuda:
            x.cuda()

        time1 = time.time()
        y = net.forward(x)
        time2 = time.time()
        print('the function took {:.3f} ms'.format((time2-time1)*1000.0))

        # Plot joints!

        return frame

    # start video stream thread, allow buffer to fill
    print("[INFO] starting threaded video stream...")
    cap = cv2.VideoCapture(0)  # default camera

    # loop over frames from the video file stream
    while True:
        # grab next frame
        _, frame = cap.read()
        key = cv2.waitKey(1) & 0xFF

        # update FPS counter
        fps.update()
        frame = predict(frame)

        # keybindings for display
        if key == ord('p'):  # pause
            while True:
                key2 = cv2.waitKey(1) or 0xff
                cv2.imshow('frame', frame)
                if key2 == ord('p'):  # resume
                    break
        cv2.imshow('frame', frame)
        if key == 27:  # exit
            break


if __name__ == '__main__':
    cuda = not args.no_cuda and torch.cuda.is_available()

    net = MyNet()
    net.load_state_dict(torch.load(args.model))
    if cuda:
        net.cuda()
    
    fps = FPS().start()
    cv2_demo(net.eval())
    fps.stop()

    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # cleanup
    cv2.destroyAllWindows()