from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
import pickle, h5py
import cv2
import torch
import torch
from torch.autograd import Variable
import json
import skimage
# from skimage.transform import resize
# import skimage as s
import pdb
import random
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)
from torchvision import transforms

from skimage import img_as_bool
from PIL import Image
from io import BytesIO
from scipy.ndimage import zoom as scizoom

from wand.image import Image as WandImage
from wand.api import library as wandlibrary
import wand.color as WandColor
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)


try:
    import accimage
except ImportError:
    accimage = None


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / zoom_factor))

    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top:trim_top + h, trim_top:trim_top + h]

import ctypes

# Tell Python about the C method
wandlibrary.MagickMotionBlurImage.argtypes = (ctypes.c_void_p,  # wand
                                              ctypes.c_double,  # radius
                                              ctypes.c_double,  # sigma
                                              ctypes.c_double)  # angle


# Extend wand.image.Image class to include method signature
class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


        
def motion_blur(x, severity=1):
    c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]

    output = BytesIO()
    x.save(output, format='PNG')
    x = MotionImage(blob=output.getvalue())

    x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

    x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8),
                     cv2.IMREAD_UNCHANGED)

    if x.shape != (256, 256):
        return Image.fromarray(np.uint8(np.clip(x[..., [2, 1, 0]], 0, 255)))  # BGR to RGB
    else:  # greyscale to RGB
        return Image.fromarray(np.uint8(np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)))

    
def zoom_blur(x, severity=1):
    c = [np.arange(1, 1.11, 0.01),
         np.arange(1, 1.16, 0.01),
         np.arange(1, 1.21, 0.02),
         np.arange(1, 1.26, 0.02),
         np.arange(1, 1.31, 0.03)][severity - 1]

    x = (np.array(x)/255.0).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)

    x = (x + out) / (len(c) + 1)
    return Image.fromarray(np.uint8(np.clip(x, 0, 1)*255.0))

def impulse_noise(x, severity=2):
    c = [.03, .06, .09, 0.17, 0.27][severity - 1]

    x = skimage.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    return Image.fromarray(np.uint8(np.clip(x, 0, 1) * 255))

def shot_noise(x, severity=4):
    c = [250, 100, 50, 30, 15][severity - 1]

    x = np.array(x) / 255.
    return Image.fromarray(np.uint8(np.clip(np.random.poisson(x * c) / c, 0, 1) * 255))

def gaussian_noise(x, severity=1):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]

    x = np.array(x) / 255.
    return  Image.fromarray(np.uint8(np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255))

def defocus_blur( x, severity=1):
        c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]

        x = np.array(x)/255.0
        kernel = disk(radius=c[0], alias_blur=c[1])
# *255
        channels = []
        for d in range(3):
            channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
        channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3

        return  Image.fromarray(np.uint8(np.clip(channels, 0, 1)*255))



def jpeg_compression(x, severity=1):
    c = [25, 18, 15, 10, 7][severity - 1]

    output = BytesIO()
    x.save(output, 'JPEG', quality=c)
    x = Image.open(output)

    return x

def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)





class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """
    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        alpha = 4
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list

class MotionBlur(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """
    def __init__(self, sev=1):
        self.sev = sev
        super().__init__()

    def forward(self, frames):
        frames2 = frames.permute(1, 0, 2,3 )
        t1 = transforms.ToPILImage()
        t2 = transforms.ToTensor()
        v = []
        for f in range(frames2.size()[0]):
            tmp = self.motion_blur(t1(frames2[f]), self.sev)
            v.append(t2(tmp))
        frames_list = torch.stack(v,0)
        frames_list = frames_list.permute(1, 0, 2,3 )
        return frames_list
    
    def motion_blur(self, x, severity=1):
        c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]

        output = BytesIO()
        x.save(output, format='PNG')
        x = MotionImage(blob=output.getvalue())

        x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

        x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8),
                         cv2.IMREAD_UNCHANGED)

        if x.shape != (224, 224):
            return Image.fromarray(np.uint8(np.clip(x[..., [2, 1, 0]], 0, 255)))  # BGR to RGB
        else:  # greyscale to RGB
            return Image.fromarray(np.uint8(np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)))

        
def gaussian_noise(x, severity=1):
        c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]
        
        x = np.array(x) / 255.
        return  Image.fromarray(np.uint8(np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255))


def rotate(x,deg):
    return x.rotate(deg)

    