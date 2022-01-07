from mmdet.apis import show_result_pyplot, init_detector, inference_detector

config_file = '/data/yunjin/last/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = '/data/yunjin/last/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
device = 'cuda:0'
# init a detector
model = init_detector(config_file, checkpoint_file, device=device)
# inference the demo image
result = inference_detector(model, '/data/yunjin/last/mmdetection/demo/demo.jpg')
show_result_pyplot(model,'/data/yunjin/last/mmdetection/demo/demo.jpg',result,0.3)