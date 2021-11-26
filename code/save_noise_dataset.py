import cv2
import numpy as np
# im
import skimage
import os
import random
from PIL import Image
from argparse import ArgumentParser
from skimage.filters import gaussian
import ffmpeg
from scipy.ndimage import zoom as scizoom

def gaussian_blur(x, severity=1):
    c = [.5, .75, 1, 1.25, 1.5][severity - 1]
    temp = x
    x = gaussian(np.array(x) / 255., sigma=c, multichannel=True)
    return Image.fromarray(np.uint8(np.clip(x, 0, 1) * 255 ))

    
def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    w = img.shape[1]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / zoom_factor))
    ch2 = int(np.ceil(w / zoom_factor))
    
    top = (h - ch) // 2
    top2 = (w - ch2) // 2
    img = scizoom(img[top:top + ch, top2:top2 + ch2], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2
    trim_top2 = (img.shape[1] - w) // 2
    return img[trim_top:trim_top + h, trim_top2:trim_top2 + w]


def zoom_blur(x, severity=1):
    c = [np.arange(1, 1.11, 0.01),
         np.arange(1, 1.16, 0.01),
         np.arange(1, 1.21, 0.02),
         np.arange(1, 1.26, 0.02),
         np.arange(1, 1.31, 0.03)][severity - 1]

    x = (np.array(x) / 255.).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)

    x = (x + out) / (len(c) + 1)
    return Image.fromarray(np.uint8(np.clip(x, 0, 1) * 255))

def impulse_noise(x, severity=2):
    c = [.03, .06, .09, 0.17, 0.27][severity - 1]
    temp=x
    x = skimage.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    return Image.fromarray(np.uint8(np.clip(x, 0, 1) * 255))

def shot_noise(x, severity=4):
    c = [250, 100, 50, 30, 15][severity - 1]
    temp=x

    x = np.array(x) / 255.
    return Image.fromarray(np.uint8(np.clip(np.random.poisson(x * c) / c, 0, 1) * 255))

def gaussian_noise(x, severity=1):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]
    temp=x

    x = np.array(x) / 255.
    return  Image.fromarray(np.uint8(np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255))

def speckle_noise(x, severity=1):
    c = [.15, .2, 0.25, 0.3, 0.35][severity - 1]

    x = np.array(x) / 255.
    return Image.fromarray(np.uint8(np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255))

def create_codec2(vidpath1,vidpath2, severity=1):

    os.system("ffmpeg -i {} -c:v mpeg4 -q:v {} {}".format(vidpath1,
                                                    15*(int)(severity),
                                                    vidpath2))

def create_codec1(vidpath1,vidpath2, severity=1):
    os.system("ffmpeg -i {} -c:v mpeg2video -q:v {} -c:a mp2 -f vob {} ".format(vidpath1,20*(int)(severity),vidpath2))


def salt_blur(x,severity=1):
    im1 = np.array(x)
    mask = np.random.randint(0,100,im1.shape)
    im2 = np.where(mask<severity*10,255,im1)
    return  Image.fromarray(np.uint8(np.clip(im2,0,255)))

def pepper_blur(x,severity=1):
    im1 = np.array(x)
    mask = np.random.randint(0,100,im1.shape)
    im2 = np.where(mask<severity*10,0,im1)
    return  Image.fromarray(np.uint8(np.clip(im2,0,255)))

def arg_parser():
    argparser = ArgumentParser(prog='noise_dataset', 
        description='Code to save noise dataset')

    argparser.add_argument('--method', type=str, default="gauss", 
    help='noise method')

    argparser.add_argument('--sev', type=int, default=5,
    help='severity level 1-5')

    argparser.add_argument('--index', type=int, default=5,
    help='severity level 1-10')
    

    args = argparser.parse_args()
    return args

args = arg_parser()
os.system("cp ./data/kinetics400/test.csv ./data/kinetics400/test.txt")
class_file = "./data/kinetics400/test.txt"
with open(class_file) as f:
   x = f.readlines()
args.index = int(args.index)

f = [line.strip() for line in x]
start = int(len(f)*(args.index-1)/15)
end = int(len(f)*(args.index)/15)
# print(len(f))

f = f[start:end]
# print(len(f))
root = os.path.join( os.environ['PT_DATA_DIR'] , "kinetics-dataset/")
noise_dir = os.path.join( os.environ['PT_DATA_DIR'] , args.method +"_t2_"+ str(args.sev))

if os.path.exists(noise_dir):
    pass
else:
    os.mkdir(noise_dir)

fn_dict = {"gauss":gaussian_noise, "shot":shot_noise, "impulse":impulse_noise, "gauss_blur":gaussian_blur,
    "zoom":zoom_blur, "speckle":speckle_noise, "salt":salt_blur, "pepper":pepper_blur}

vidcount = 0
# for row in 
# for vid in os.listdir(root):
for vid in f:
# for i in range(1):
    # vidpath = os.path.join(root,vid)
    # if vid[len(vid)-4:len(vid)] == ".mp4":
    #     continue
    vid = vid.split(',')[0].split('/')[-1]
    if args.method == "codec1":
        oldpath = os.path.join(root, vid)
        newpath = os.path.join(noise_dir ,vid[:-4]+'.mp4')
        create_codec1(oldpath,newpath,args.sev)
        continue

    elif args.method == "codec2":
        oldpath = os.path.join(root, vid)
        newpath = os.path.join(noise_dir ,vid[:-4]+'.avi')
        create_codec2(oldpath,newpath,args.sev)
        continue
        
    vidcap = cv2.VideoCapture(os.path.join(root, vid))
    # vid_path = "example.mp4"
    # vidcap = cv2.VideoCapture(vid_path)

    fps = vidcap.get(cv2.CAP_PROP_FPS)
    # info = ffmpeg.probe(os.path.join(root, vid))
    # vs = next(c for c in info['streams'] if c['codec_type'] == 'video')
    # duration = float(vs['duration'])
    # fps = vs['avg_frame_rate']
    # fps = fps.split('/')
    # fps = int(fps[0])/int(fps[1])
    # vid = vid_path

    success,image = vidcap.read()
    height, width, layers = image.shape
    size = (width,height)
    p = os.path.join(noise_dir ,vid[:-4]+'.mp4')
    if os.path.exists(p):
        vidcap.release()
        continue
    out = cv2.VideoWriter( os.path.join(noise_dir ,vid[:-4]+'.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    # out = cv2.VideoWriter( "noise.mp4", cv2.VideoWriter_fourcc(*'MP4V'), fps, size)

    while success:
        # ret, frame = cap.read()
        noise = fn_dict[args.method](Image.fromarray(image), args.sev)
        # if args.method == "gauss_blur2":
        #     pass
        # else:
        noise = np.array(noise)
        success,image = vidcap.read()
        out.write(noise)
        # if ret==True:
        #     frame = cv2.flip(frame,0)

        #     # write the flipped frame
        #     out.write(frame)

        #     cv2.imshow('frame',frame)impulse_noise
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        # else:
        #     break
    vidcap.release()
    out.release() 
    vidcount=vidcount+1
    print(str(vidcount)+" videos processed")