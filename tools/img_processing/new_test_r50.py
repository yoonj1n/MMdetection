######For uncrop first img#######
####img path ->"/data/yunjin/last/salmon"

import cv2
import os, sys
import numpy as np
import matplotlib.pyplot as plt


IMAGE_PATHS = "/data/khyunjung/tf_sal/images/test"
SAVE_PATHS = "/data/yunjin/last/mmdetection/salmonimg/test_r50"

def perspective_correction(src, max_height=790, max_width=790):
   # Get color ranges at the corners
   height, width = src.shape[:2]
   img_height = (int)(height*0.7)
   img_width = (int)(width*0.7)
#    print(src[height-5,width-5].tolist())
   color = []
   off_pixel = 10
   color.append(src[off_pixel, off_pixel].tolist())
   color.append(src[height-off_pixel, off_pixel].tolist())
   color.append(src[off_pixel, width-off_pixel].tolist())
   color.append(src[height-off_pixel, width-off_pixel].tolist())
#    print(color)
   # lower and upper ranges
   lower = []
   upper = []
   for i in range (3):
      color.sort(key=lambda k: k[i])   # sort 가장 낮은 수 부터 높은 수 까지 -> Range 정하기
    #   print(color)
      low_col = color[0][i]-10
      if low_col < 0:
         lower.append(0)
      else:
         lower.append(low_col)

      up_col = color[3][i]+10
      if up_col > 255:
         upper.append(255)
      else:
         upper.append(up_col)
#    print(lower)
#    print(upper)
   lower = np.array(lower, dtype="uint8")
   upper = np.array(upper, dtype="uint8")
   mask = cv2.inRange(src, lower, upper)
   bound = cv2.bitwise_and(src, src, mask=mask)
#    plt.imshow(cv2.cvtColor(bound, cv2.COLOR_BGR2RGB)), plt.show() #range를 토대로 mask 형성

   gray = cv2.cvtColor(bound, cv2.COLOR_BGR2GRAY) #gray 마스크로 변경
# #    plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)), plt.show()
   blur = cv2.GaussianBlur(gray, (7,7), 0)
   

#    # detect the corners
   corners = cv2.goodFeaturesToTrack(blur, 4, 0.01, 500, blockSize = 15)  # four points
   corners = np.int0(corners)
#    print(corners)


   if len(corners) == 4:
      srcPoints = []
      for i in corners:
         x, y = i.ravel()
         srcPoints.append([x, y])
#          #cv2.circle(src, (x, y), 5, (0,0,255), -1)

      # Order four points
      srcPoints.sort(key=lambda k: k[0]) # sort by x
      #print(srcPoints)
      first_col = []
      first_col.append(srcPoints[0])
      first_col.append(srcPoints[1])
      first_col.sort(key=lambda k: k[1])

      second_col = []
      second_col.append(srcPoints[2])
      second_col.append(srcPoints[3])
      second_col.sort(key=lambda k: k[1])

      srcPoints = []
      srcPoints.append(first_col[0])
      srcPoints.append(first_col[1])
      srcPoints.append(second_col[0])
      srcPoints.append(second_col[1])
      
    #   for i in range(4):
    #       tmp = srcPoints[i][0]
    #       srcPoints[i][0]=srcPoints[i][1]
    #       srcPoints[i][1]=tmp
    #   print(np.float32(srcPoints))
    #   plt.imshow(blur), plt.show()

#       # align with findHomography
      
      dstPoints = [[0, 0], [0, img_height], [img_width, 0], [img_width, img_height]]
      H, mask = cv2.findHomography(np.float32(srcPoints), np.float32(dstPoints))
      dst = cv2.warpPerspective(img, H, (img_width, img_height))
      # plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)), plt.show()
      return cv2.warpPerspective(src, H, (img_width, img_height))
    #   plt.imshow(mask), plt.show()
   else:
      return cv2.resize(src, (img_width, img_height), interpolation=cv2.INTER_AREA)
##############################################################################################

file_list = os.listdir(IMAGE_PATHS)


for file in file_list:
   if file.endswith(('.jpg', '.JPG', '.jpeg', 'JPEG', '.png', 'PNG')):
      src = os.path.join(IMAGE_PATHS, file)
      img = cv2.imread(src)
      result = perspective_correction(img)
      #height, width = result.shape[:2]
      #print(height, width)
      name = os.path.join(SAVE_PATHS,file)
      cv2.imwrite(name,result.copy())