# The new config inherits a base config to highlight the necessary modification
_base_ = '/data/yunjin/last/mmdetection/configs/resnest/mask_rcnn_s50_fpn_syncbn-backbone+head_mstrain_1x_coco_cafe.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)))

# optimizer
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=5000,
    warmup_ratio=0.001,
    min_lr=0.0025,
    step=[8, 16])
total_epochs = 600

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('Scar',)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        img_prefix='/data/lulu/mmdetection/img/train/',
        classes=classes,
        ann_file='/data/lulu/mmdetection/img/train/annotations.json'),
    val=dict(
        img_prefix='/data/lulu/mmdetection/img/val/',
        classes=classes,
        ann_file='/data/lulu/mmdetection/img/val/annotations.json'),
    test=dict(
        img_prefix='/data/yunjin/last/mmdetection/pkl/',
        classes=classes,
        ann_file='/data/yunjin/last/mmdetection/pkl/annotations.json'))

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_form = '/home/yunjin/.cache/torch/hub/checkpoints/resnest50_d2-7497a55b.pth'
