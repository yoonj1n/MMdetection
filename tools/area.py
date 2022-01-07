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
    #setting
    workpath = os.getcwd()
    count =0
    # test image
    files = os.listdir(args.img)
    #mask image 저장 공간 생성
    config_name = args.config.split('/')[-1].split('.')[0]
    maskimg_path = workpath+'/mask_img'+'/'+config_name
    os.makedirs(maskimg_path,exist_ok=True)
    
    for file in files:
        if file.endswith(('.jpg', '.JPG', '.jpeg', '.JPEG', '.png')):
            imgname = args.img+file
            result = inference_detector(model, imgname)
            
            #Mask 영역 구하기
            masks = [np.where(m == 1, 255, m) for m in result[1]]
            mask = masks[0]

            #mask 영역 사진을 얻기 위해 0,255로 이루어진 배열 구성
            tmp = np.array(mask[0]).astype(np.uint8)
            test = np.where(tmp == 1,255,tmp)

            #mask 영역 사진 구하여 저장
            mask_img = Image.fromarray(test)
            mask_img_name = file.split('.')[0]
            mask_img.save(maskimg_path+'/'+mask_img_name+'.jpg')
            
            #pixel 갯수 구하기
            for i in mask[0]:
                for k in i:
                    count+=k

            print("filename : "+imgname)
            print("Detect : ", end=' ')
            print(count, end=' ')
            print("     Total : ",end=' ')
            print(test.size)
            print(" Ratio : ",end=' ')
            print(count/(test.size))
            count =0
    

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
