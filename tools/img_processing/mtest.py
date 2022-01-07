# %%
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

#%%
#src = cv.imread('cyanCorner.jpg')  # working
src = cv.imread('cyanCorner2.jpg')   # working

# %%
# image resize target
height, width = src.shape[:2]
max_height = 700
max_width = 700
# # get scaling factor
# scaling_factor = 1.0
# if width < height:
#    scaling_factor = max_width / float(width)
# else:
#    scaling_factor = max_height / float(height)
# # resize image
# if scaling_factor < 1.0:
#    img = cv.resize(src, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv.INTER_AREA)
# else:
#    img = cv.resize(src, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv.INTER_LINEAR)
# #print(img.shape)

# %%
# color
lower = [230, 200, 0]
upper = [255, 255, 100]
lower = np.array(lower, dtype="uint8")
upper = np.array(upper, dtype="uint8")
mask = cv.inRange(img, lower, upper)
bound = cv.bitwise_and(img, img, mask=mask)

#plt.imshow(cv.cvtColor(bound, cv.COLOR_BGR2RGB)), plt.show()

gray = cv.cvtColor(bound, cv.COLOR_BGR2GRAY)
#plt.imshow(cv.cvtColor(gray, cv.COLOR_BGR2RGB)), plt.show()
blur = cv.GaussianBlur(gray, (7,7), 0)
#plt.imshow(cv.cvtColor(blur, cv.COLOR_BGR2RGB)), plt.show()

# %%
corners = cv.goodFeaturesToTrack(blur, 4, 0.01, 350)  # four points
corners = np.int0(corners)

srcPoints = []
for i in corners:
   x, y = i.ravel()
   #print(x, y)
   srcPoints.append([x, y])
   #cv.circle(img, (x, y), 5, (0,0,255), -1)

# Order four points
srcPoints.sort(key=lambda k: k[1])
srcPoints.sort(key=lambda k: k[0])
#print(srcPoints)
#plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)), plt.show()

# %%
# align with findHomography
dstPoints = [[0, 0], [0, 640], [640, 0], [640, 640]]
H, mask = cv.findHomography(np.float32(srcPoints), np.float32(dstPoints))
dst = cv.warpPerspective(img, H, (640, 640))
plt.imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB)), plt.show()

# %%
