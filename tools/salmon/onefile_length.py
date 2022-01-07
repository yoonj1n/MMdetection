import numpy as np
from PIL import Image


from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)



def main():
    config = "/data/yunjin/last/mmdetection/configs/_base_/salmon/mask_1x_salmon_r50_dataset2.py"
    checkpoint = "/data/yunjin/last/mmdetection/configs/_base_/salmon/work_dirs/mask_1x_salmon_r50_dataset2/epoch_600.pth"
    #img ="/data/yunjin/last/salmon/img/last/f12.jpg"
    img = "/data/yunjin/last/mmdetection/tools/salmon/test/many.jpg"
    # build the model from a config file and a checkpoint file
    model = init_detector(config, checkpoint)

    imgname = img
    result = inference_detector(model, imgname)

    count_1=0
    count_0=0
    count_2 =0
    #Mask 영역 구하기
    masks = [np.where(m == 1, 255, m) for m in result[1]]
    #print(masks)

    result_mask_0 =[0]
    result_mask_1 =[0]
    result_mask_2=[0]

    if len(masks[0]) != 0 : #0 : body 1 : folk 2 : total
        mask_0 = masks[0] 
        result_mask_0 = mask_0[0][3][:]
        for i in mask_0[0]:
            result_mask_0 = [x+y for x,y in zip(result_mask_0,i)]
        for k in result_mask_0:
            if k>0:
                count_0+=1
        #print("img_W : ", end=' ')
        #print(len(result_mask_0))
    #pixel= 90/len(result_mask_0)
    print(len(mask_0[1]))
    #result_mask_# = img_w, pixel로 계산 할 수 있다.

    # if len(masks[1]) != 0 : 
    #     mask_1 = masks[1] 
    #     result_mask_1 = mask_1[0][0][:]
    #     for i in mask_1[0]:
    #         result_mask_1 = [x+y for x,y in zip(result_mask_1,i)]
    #     for k in result_mask_1:
    #         if k>0:
    #             count_1+=1
    #     print(result_mask_1)

    # if len(masks[2]) != 0 : 
    #     mask_2 = masks[2] 
    #     result_mask_2 = mask_2[0][0][:]
    #     for i in mask_2[0]:
    #         result_mask_2 = [x+y for x,y in zip(result_mask_2,i)]
    #     for k in result_mask_2:
    #         if k>0:
    #             count_2+=1

    model.show_result(imgname,result,out_file='/data/yunjin/last/mmdetection/tools/salmon/test/'+"test.jpg")



if __name__ == '__main__':
    main()
