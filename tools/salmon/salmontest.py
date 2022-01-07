import asyncio
import numpy as np
import os
import cv2
import time
from PIL import Image
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Test Image path')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


#img processing code
def perspective_correction(src, max_height=790, max_width=790):
   # Get color ranges at the corners
   height, width = src.shape[:2]
   img_height = (int)(height*0.7)
   img_width = (int)(width*0.7)
#    print(src[height-5,width-5].tolist())
   color = []
   off_pixel = 1
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
      #dst = cv2.warpPerspective(img, H, (img_width, img_height))
      # plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)), plt.show()
      return cv2.warpPerspective(src, H, (img_width, img_height))
    #   plt.imshow(mask), plt.show()
   else:
      return cv2.resize(src, (img_width, img_height), interpolation=cv2.INTER_AREA)


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    
    #image file
    files = os.listdir(args.img)
    config_name = str(args.config).split('/')[-1]
    
    #Save path set -> config and model file folder
    SAVE_PATHS = str(args.config)[:(-1)*len(config_name)]
    
    #excel file make
    f=open(SAVE_PATHS + (config_name.split('.')[0])+"_length.xlsx",'w')
    f.write("IMAGE_NAME, total, R_total, fork, R_fork, body, R_body"+"\n")
    start = time.time()
    #img file load
    for file in files:
        if file.endswith(('.jpg', '.JPG', '.jpeg', '.JPEG', '.png')):
            imgname = args.img+'/'+file
            
            
            #img processing and save            
            img = cv2.imread(imgname)
            rimg = perspective_correction(img)

            crop_img_path = SAVE_PATHS+"crop_img/"
            os.makedirs(crop_img_path,exist_ok=True)
            n = os.path.join(crop_img_path,file)
            
            cv2.imwrite(n,rimg.copy())

            result = inference_detector(model, n)

            count_1=0
            count_0=0
            count_2 =0
            #Mask 영역 구하기
            masks = [np.where(m == 1, 255, m) for m in result[1]]
            


            result_mask_0 =[0]
            result_mask_1 =[0]
            result_mask_2=[0]

            if len(masks[0]) != 0 : #0 : body 1 : folk 2 : total
                mask_0 = masks[0] 
                result_mask_0 = mask_0[0][0][:]
                #print(result_mask_0)
                for i in mask_0[0]:
                    result_mask_0 = [x+y for x,y in zip(result_mask_0,i)]
                for k in result_mask_0:
                    if k>0:
                        count_0+=1
                #print("img_W : ", end=' ')
                #print(len(result_mask_0))
            pixel= 89.9/len(result_mask_0)
            #result_mask_# = img_w, pixel로 계산 할 수 있다.
            
            if len(masks[1]) != 0 : 
                mask_1 = masks[1] 
                result_mask_1 = mask_1[0][0][:]
                for i in mask_1[0]:
                    result_mask_1 = [x+y for x,y in zip(result_mask_1,i)]
                for k in result_mask_1:
                    if k>0:
                        count_1+=1

            if len(masks[2]) != 0 : 
                mask_2 = masks[2] 
                result_mask_2 = mask_2[0][0][:]
                for i in mask_2[0]:
                    result_mask_2 = [x+y for x,y in zip(result_mask_2,i)]
                for k in result_mask_2:
                    if k>0:
                        count_2+=1
            
            filename=imgname.split('/')[-1]

            f.write(str(filename)+','+str(count_2)+','+str(pixel*count_2)+','+str(count_1)+','+str(pixel*count_1)+','+str(count_0)+','+str(pixel*count_0)+'\n')

            #img save
            saveimg_path = SAVE_PATHS+'/result_img'
            os.makedirs(saveimg_path,exist_ok=True)

            model.show_result(rimg,result,out_file=saveimg_path+'/'+str(file).split('.')[0]+'.jpg')
    print(time.time()-start)
    f.close()

    print("***************************DONE***************************")
    

async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(model, args.img, result[0], score_thr=args.score_thr)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
