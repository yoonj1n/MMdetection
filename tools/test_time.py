import asyncio
import numpy as np
import time
import os
from PIL import Image
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Test Image Path')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('repeat', help='Test repeat number')
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
    #test path
    files = os.listdir(args.img)    
    #count = 0
    #max = 0

    #test time file set
    with open(os.getcwd()+"/valtime.csv",'a') as f:
        f.write("\n")
        f.write("\n")
        f.write(str(args.config).split('/')[-1])
        
        for max in range(int(args.repeat)):
            #validation start
            start = time.time()
            for file in files:
                if file.endswith(('.jpg', '.JPG', '.jpeg', '.JPEG', '.png')):
                    imgname = args.img+file
                    #count+=1
                    result = inference_detector(model, imgname)
                    #print(count,end = ' ')
            #validation done / print validation time
            f.write("\n")
            f.write(str(time.time()-start))
        
            
    
    


async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
