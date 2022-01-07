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
    parser.add_argument('--out', type=str, default=None)
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
    # test image
    files = os.listdir(args.img)
    #csv 파일 열기
    f=open(args.out,'w')
    #f.write(args.out)
    #헤더쓰기
    f.write("IMAGE_NAME, total_pixel, mask_pixel, rate"+"\n")
    #setting
    workpath = os.getcwd()
    count =0
    #mask image 저장 공간 생성
    config_name = args.config.split('/')[-1].split('.')[0]
    maskimg_path = workpath+'/mask_img'+'/'+config_name
    os.makedirs(maskimg_path,exist_ok=True)
    #열
    '''IMAGE_NAME=list(range(len(files)))
    total_pixel =list(range(len(files)))
    mask_pixel=list(range(len(files)))
    rate=list(range(len(files)))'''
    #리스트 길이 수동으로 지정. 아니면 files에서 이미지가 몇개인지 구하기
    IMAGE_NAME=list(range(80))
    total_pixel =list(range(80))
    mask_pixel=list(range(80))
    rate=list(range(80))

    j=0
    for file in files:
        if file.endswith(('.jpg', '.JPG', '.jpeg', '.JPEG', '.png')):
            imgname = args.img+file
            result = inference_detector(model, imgname)
            #mask 영역 구하기
            masks = [np.where(m == 1, 255, m) for m in result[1]]
            mask = masks[0]

            #mask 영역 사진을 얻기 위해 0,255로 이루어진 배열 구성
            tmp = np.array(mask[0]).astype(np.uint8)
            test = np.where(tmp == 1,255,tmp)

            #mask 영역 사진 구하여 저장
            mask_img = Image.fromarray(test)
            mask_img_name = file.split('.')[0]
            mask_img.save(maskimg_path+'/'+mask_img_name+'.jpg')
            #test = np.array(mask)

            for i in mask[0]:
                for k in i:
                    count+=k

            filename=imgname.split('/')[-1]

            IMAGE_NAME[j]=filename
            total_pixel[j]=test.size
            ratio=count/(test.size)
            mask_pixel[j]=count
            rate[j]=ratio
            j=j+1
                       
            '''print("IMGNAME : ", end=' ')
            print(filename)
            IMAGE_NAME[j]=filename
            print("Detect : ", end=' ')
            print(count, end=' ')
            mask_pixel[j]=count
            print("     Total : ",end=' ')
            print(test.size)
            total_pixel[j]=test.size
            print(" Ratio : ",end=' ')
            print(count/(test.size))
            rate[j]=count/(test.size)'''

            count =0

    #for i in range(len(mask_pixel)) :
        #print(i)

    #숫자->문자 형변환
    for i in range(len(mask_pixel)):
        f.write(str(IMAGE_NAME[i])+','+str(total_pixel[i])+','+str(mask_pixel[i])+','+str(rate[i])+'\n')

    f.close()
    '''if args.out is None:
        plt.show()
    else:
        print(f'save txt to: {args.out}')
        plt.savefig(args.out)
        plt.cla()'''
    
    """
    for m in result[1]:
        m = np.array(m)
        print(m.shape)
        m = m.reshape(-1,1184)
        print(m.shape)
        mask_imgs = [Image.fromarray(m,'1')]
        [m.save(f'{i}.png') for i, m in enumerate(mask_imgs)]
    """
    

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
