#!/usr/bin/env bash

# First edit models/my_ssd_mobilenet_v2_fpnlite/pipeline.config
# num_classes, batch_size, fine_tune_checkpoint, num_steps if needed
# num_steps: 20000 are recommended at first
# If the final loss <0.1 or ~0.1, then stop, if not then increase num_steps and modify pipeline.config ("fine_tune_checkpoint" to the last one in the models/my_ssd_mobilenet_v2_fpnlite/ckpt-??), and run this script again

# nohup ./train.sh &


python /data/yunjin/last/condor/test.py
