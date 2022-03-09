import csv
import numpy as np
import pickle

import cv2
import math
import random

import os


def translation(x,sev,nfdict):
    dic2 = {}
    for row in x:
        p = row.split(',')[0]
        vid = p.split('/')[-2] + "/" + p.split('/')[-1]
        dic2[vid] = [(random.randint(0,sev*8),random.randint(0,sev*8)) for i in range(350)]
    
    return dic2

def random_rotation(x,sev,nfdict):
    dic2 = {}
    for row in x:
        p = row.split(',')[0]
        vid = p.split('/')[-2] + "/" + p.split('/')[-1]
        dic2[vid] = [random.randint(-sev*6-1,sev*6+1) for i in range(350)]
    
    return dic2


def jumbling(x,sev,nfdict):
    dic2 = {}
    seg = 64/(2**sev)
    for row in x:
        final = []
        p = row.split(',')[0]
        total = int(nfdict[vid])
        vid = p.split('/')[-2] + "/" + p.split('/')[-1]
        for j in range(math.ceil(total/seg)):    
             tmp = list(range(j*seg,min((j+1)*seg,total)))
             random.shuffle(tmp)
             final += tmp
        dic2[vid] = final
    return dic2

def box_jumbling(x,sev,nfdict):
    dic2 = {}
    seg = (sev+1)**2
    for row in x:
        p = row.split(',')[0]
        vid = p.split('/')[-2] + "/" + p.split('/')[-1]
        total = int(nfdict[vid])
        final = []
        for j in range(math.ceil(total/seg)):    
             tmp = list(range(j*seg,min((j+1)*seg,total)))
             
             final.append(tmp)
        random.shuffle(final)
        dic2[vid] = sum(final, [])
    return dic2

def freezing(x,sev,nfdict):
    sev =  0.12*sev
    dic2 = {}
    for row in x:
        p = row.split(',')[0]
        vid = p.split('/')[-2] + "/" + p.split('/')[-1]
        total = int(nfdict[vid])
        final = [0]
        x=1
        while x<len(total):
            if random.random()<sev:
                final.append(final[-1])
            else:
                final.append(x)
            x = x + 1
    return dic2


class_file = "./kinetics400_val_nf.txt"
class_file = "./ucf101_val_nf.txt"
class_file = "./hmdb51_val_nf.txt"
with open(class_file) as f:
    x = f.readlines()

nfdict = {}
for row in x:
    nfdict[row.split(',')[0]] = row.split(',')[1]

class_file = "./data/hmdb51/test.txt"
# class_file = "./data/ucf101/test.txt"
# class_file = "./data/kinetics400/test.txt"
with open(class_file) as f:
    x = f.readlines()

    

sampling_rate = 5
nf=16
dic2={}
sev=4

class_file = "./data/hmdb51/test.txt"
# class_file = "./data/kinetics400/test.txt"
with open(class_file) as f:
    x = f.readlines()

methods = {"random_rotation":random_rotation, "translation":translation, 
"jumbling":jumbling, "box_jumbling":box_jumbling, "freezing":freezing}
for method,fn in methods:
    for sev in range(5):

        dic2 = methods[method](x,sev,nfdict)
        with open('./dict_hmdb/'+method+'_'+str(sev)+'.pickle', 'wb') as handle:
            pickle.dump(dic2, handle, protocol=pickle.HIGHEST_PROTOCOL)


