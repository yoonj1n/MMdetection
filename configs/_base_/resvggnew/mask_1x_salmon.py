# The new config inherits a base config to highlight the necessary modification
_base_ = '/data/yunjin/last/mmdetection/configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=3),
        mask_head=dict(num_classes=3)))

# optimizer
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=5000,
    warmup_ratio=0.001 ,
    min_lr=0.0025,
    step=[8, 16])
total_epochs = 600

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('total','folk','body')
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        img_prefix='/data/khyunjung/mm_sal/annotation/images/train/',
        classes=classes,
        ann_file='/data/khyunjung/mm_sal/annotation/images/train/annotations.json'),
    val=dict(
        img_prefix='/data/khyunjung/mm_sal/annotation/images/val/',
        classes=classes,
        ann_file='/data/khyunjung/mm_sal/annotation/images/val/annotations.json'),
    test=dict(
        img_prefix='/data/khyunjung/mm_sal/annotation/images/test/',
        classes=classes,
        ann_file='/data/khyunjung/mm_sal/annotation/images/test/annotations.json'))

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_form = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_1x_coco/mask_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.38__segm_mAP-0.344_20200504_231812-0ebd1859.pth'
