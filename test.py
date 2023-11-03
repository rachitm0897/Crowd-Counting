import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy
import json
import torchvision.transforms.functional as F
from matplotlib import cm as CM
from image import *
from model import CSRNet
import torch
from matplotlib import cm as c
import cv2
from torchvision import datasets, transforms
'''
'''
transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])

model = CSRNet()

#defining the model

model = model.cuda()

#loading the trained weights

checkpoint = torch.load(r'C:\Users\91987\Documents\python\Crowd\CSRnet_python_3\crowd_analysis\ckpts\model-76.74.pth.tar')

model.load_state_dict(checkpoint['state_dict'])

img_path = r"C:\Users\91987\Downloads\photo1.jpg"

img = transform(Image.open(img_path).convert('RGB')).cuda()

'''
'''
output = model(img.unsqueeze(0))

print("Predicted Count : ", int(output.detach().cpu().sum().numpy()))

temp = np.asarray(output.detach().cpu().reshape(output.detach().cpu().shape[2],output.detach().cpu().shape[3]))

plt.imshow(temp, cmap = c.jet)

plt.show()

# temp = h5py.File(r'C:\Users\91987\Documents\python\Crowd\Shanghai\part_A_final\test_data\ground_truth\IMG_92.h5', 'r')

# temp_1 = np.asarray(temp['density'])

# plt.imshow(temp_1,cmap = c.jet)

# print("Original Count : ",int(np.sum(temp_1)) + 1)

# plt.show()

# print("Original Image")

# plt.imshow(plt.imread(r'C:\Users\91987\Documents\python\Crowd\Shanghai\part_A_final\test_data\images\IMG_92.jpg'))

# plt.show()


# cam = cv2.VideoCapture(0)

# while True:
#     ignore, frame = cam.read()
#     frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#     img = transform(frame).cuda()
#     output = model(img.unsqueeze(0))
#     print("Predicted Count : ", int(output.detach().cpu().sum().numpy()))
#     temp = np.asarray(output.detach().cpu().reshape(output.detach().cpu().shape[2],output.detach().cpu().shape[3]))
#     cv2.imshow("Heatmap",temp)
#     frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
#     cv2.imshow("Original",frame)

#     if cv2.waitKey(1) & 0xff == ord('q'):
#         break




