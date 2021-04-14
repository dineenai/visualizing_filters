#!/bin/bash


PYTHON="/opt/anaconda3/envs/blurry_vision/bin/python"
RESUME="/data/blurry_vision_sup_RN50/supervised_resnet50_conv1_21_from_gauss_0_for_30_epoch_to_gauss_4_for_30/outmodel/checkpoint_supervised_resnet50_conv1_21_from_gauss_0_for_30_epoch_to_gauss_4_for_30_epoch35.pth.tar"
FFILE="sup_RN50_conv1_21_from_gauss_0_for_30_epoch_to_gauss_4_for_30_epoch35"
ARCH='resnet50'

${PYTHON} Visualiser_rf_21.py --resume ${RESUME} \
--a ${ARCH} \
--save_filter_file ${FFILE}
