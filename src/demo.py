import numpy as np
import torch
import cv2
import argparse
import time
from imutils.video import FPS, WebcamVideoStream
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='Live Demo')
parser.add_argument('--model', default='../models/nn.model',
                    type=str, help='Trained model path')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

def cv2_demo(net):
    def predict(frame):
        frame = cv2.resize(frame, (320, 320)).astype(np.float32)
        frame = frame.astype(np.float32)

        x = torch.from_numpy(frame).permute(2,0,1)
        x = Variable(x.unsqueeze(0))
        y = net(x).data

        # Plot joints!

        return frame

    # start video stream thread, allow buffer to fill
    print("[INFO] starting threaded video stream...")
    stream = WebcamVideoStream(src=0).start()  # default camera
    time.sleep(2.0)
    # start fps timer
    # loop over frames from the video file stream
    while True:
        # grab next frame
        frame = stream.read()
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
    net = torch.load(args.model, map_location='cpu')
    if args.cuda:
      net.cuda()

    fps = FPS().start()
    cv2_demo(net.eval())
    fps.stop()

    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # cleanup
    cv2.destroyAllWindows()