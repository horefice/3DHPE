import numpy as np
import torch
import cv2 as cv
import datetime
import os

class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

def writeArgsFile(args,saveDir):
  args_list = dict((name, getattr(args, name)) for name in dir(args)
                if not name.startswith('_'))
  file_name = os.path.join(saveDir, 'args.txt')
  with open(file_name, 'a') as opt_file:
    opt_file.write('\n==> Args ('+datetime.datetime.now().isoformat()+'):\n')
    for k, v in sorted(args_list.items()):
       opt_file.write('  {}: {}\n'.format(str(k), str(v)))

# def extract_2d_joints_from_heatmap(heatmap, input_size):
#   num_joints = heatmap.shape[2]
#   joints_2d = np.zeros(shape=(num_joints, 2), dtype=np.int32)
#   heatmap_resized = cv.resize(heatmap, (input_size, input_size))

#   for joint_num in range(num_joints):
#       joint_coord = np.unravel_index(np.argmax(heatmap_resized[:, :, joint_num]), (input_size, input_size))
#       joints_2d[joint_num, :] = joint_coord

#   return joints_2d

# def extract_3d_joints_from_heatmap(joints_2d, x_hm, y_hm, z_hm, input_size):
#   num_joints = joints_2d.shape[0]
#   joints_3d = np.zeros(shape=(num_joints, 3), dtype=np.float32)
#   for joint_num in range(num_joints):
#       coord_2d_y = joints_2d[joint_num][0]
#       coord_2d_x = joints_2d[joint_num][1]

#       joint_x = x_hm[max(int(coord_2d_x/8), 1), max(int(coord_2d_y/8), 1), joint_num] * 10
#       joint_y = y_hm[max(int(coord_2d_x/8), 1), max(int(coord_2d_y/8), 1), joint_num] * 10
#       joint_z = z_hm[max(int(coord_2d_x/8), 1), max(int(coord_2d_y/8), 1), joint_num] * 10
#       joints_3d[joint_num, 0] = joint_x
#       joints_3d[joint_num, 1] = joint_y
#       joints_3d[joint_num, 2] = joint_z
#   joints_3d -= joints_3d[14, :]

#   return joints_3d.reshape(-1)

# def heatmaps_to_3d_joints(net_output, input_size):
#   joints_3d = np.zeros(shape=(net_output.size(0),net_output.size(-1)*3), dtype=np.float32)
#   for idx, out in enumerate(net_output):
#     heatmap = out[0].detach().numpy()
#     x_hm, y_hm, z_hm = out[1:].detach().numpy()
#     joints_2d = extract_2d_joints_from_heatmap(heatmap, input_size)
#     joints_3d[idx] = extract_3d_joints_from_heatmap(joints_2d, x_hm, y_hm, z_hm, input_size)

#   return torch.from_numpy(joints_3d)
