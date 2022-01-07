import asyncio
import numpy as np
import os
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


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    
    #image file
    files = os.listdir(args.img)
    config_name = str(args.config).split('/')[-2]
    #csv file
    f=open(config_name+"_length.csv",'w')
    f.write("IMAGE_NAME, total, R_total, fork, R_fork, body, R_body"+"\n")

    for file in files:
        if file.endswith(('.jpg', '.JPG', '.jpeg', '.JPEG', '.png')):
            imgname = args.img+file
            result = inference_detector(model, imgname)

            count_1=0
            count_0=0
            count_2 =0
            #Mask 영역 구하기
            masks = [np.where(m == 1, 255, m) for m in result[1]]

            if len(masks[0]) != 0 : #0 : body 1 : folk 2 : total
                mask_0 = masks[0] 
                result_mask_0 = mask_0[0][0][:]
                for i in mask_0[0]:
                    result_mask_0 = [x+y for x,y in zip(result_mask_0,i)]
                for k in result_mask_0:
                    if k>0:
                        count_0+=1
                #print("img_W : ", end=' ')
                #print(len(result_mask_0))
            pixel= 90/len(result_mask_0)
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
            '''
            print("total : ", end=' ')
            print(count_0)
            print("folk : ", end=' ')
            print(count_1)
            print("body : ", end=' ')
            print(count_2)
            print("real total length : ", end=' ')
            print(pixel*count_0)
            '''

            filename=imgname.split('/')[-1]

            f.write(str(filename)+','+str(count_2)+','+str(pixel*count_2)+','+str(count_1)+','+str(pixel*count_1)+','+str(count_0)+','+str(pixel*count_0)+'\n')

    f.close()
    # print(result_mask_0 == result_mask_1)
    

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
