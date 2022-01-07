#!/usr/bin/env python

# First, all images and xml should be in train/ directory 
# randomly shuffle image and xml data and split 20%
# Then move them to test/ directory

import os
import shutil
import random
import argparse

#parse
parser = argparse.ArgumentParser()
parser.add_argument('W', type=str, help = "Enter test OR val")

args = parser.parse_args()
#cwd = os.getcwd()
rpath = "/data/yunjin/last/mmdetection/configs/_base_/scar3/train/"
wpath = "/data/yunjin/last/mmdetection/configs/_base_/scar3/"+args.W+"/"
files = os.listdir(rpath)

jpgList = []
for file in files:
   if file.endswith(('.jpg', '.JPG', '.jpeg', '.JPEG', '.png')):
      jpgList.append(file)
#print(jpgList[0].split('.')[0])

random.shuffle(jpgList)
testList = jpgList[0:int(0.2*len(jpgList))] # 20%
#testList = jpgList[0:180] # the number as 20% of all images

# move files
for file in testList:
   shutil.move(rpath+file, wpath+file) # images files
   name, ext = os.path.splitext(file)
   shutil.move(rpath+name+'.json', wpath+name+'.json') # json files