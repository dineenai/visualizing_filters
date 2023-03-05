# 23rd of Feb 2022
#  source activate blurry_vision
# https://colab.research.google.com/drive/1FG7vegjOjeHpkStBvpJEcSeumfwLXD1w#scrollTo=jmcIN_cdTBa_
# ammended from Copy of gitGabs2_Aine_Edited.ipynb
# Code from Alex Wade - Nov 21

import numpy as np
import matplotlib.pyplot as plt

# from dog_gen import GaborGenerator,DOGGenerator,trainer_fn, trainer_fn2
from utils import dog_fn,gabor_fn
import matplotlib.image as img  
import cv2

# source of np arrays is: /home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/visualizing_filters/
#mv /home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/visualizing_filters/fit_rfs_23222.py /home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/fit_receptive_field