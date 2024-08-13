# Print Single RFs....

# Test Command:
# python3 Visualiser_rf_21_single_field_test.py --a  'resnet50' --resume "/data/blurry_vision_sup_RN50/supervised_resnet50_conv1_21_gauss_0_for_60_epoch/outmodel/checkpoint_supervised_resnet50_conv1_21_gauss_0_for_60_epoch_epoch60.pth.tar"

# bash script: filters.sh 
# Importing libraries
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as t
import cv2 as cv
# import torchvision.models as models
import resnet_conv1_21 as models #Copy this file to pwd
# Importing the module
from extractor import Extractor
from collections import OrderedDict #OrderedDict()

import argparse
import os
import torch.backends.cudnn as cudnn

import numpy as numpy

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')               # 
# parser.add_argument('--save_filter_path', default='/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/visualizing_filters/receptive_fields_V1',
#                     type=str, metavar='SAVE_FILTER_PATH',
#                     help='path to save accuracy of model') 
# # TEST 
# parser.add_argument('--save_filter_path', default='/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/visualizing_filters/test_single_rfs',
#                     type=str, metavar='SAVE_FILTER_PATH',
#                     help='path to save accuracy of model') 
parser.add_argument('--save_filter_path', default='/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/visualizing_filters/conv1_filters',
                    type=str, metavar='SAVE_FILTER_PATH',
                    help='path to save accuracy of model') 
# model name parser.....

parser.add_argument('--net_name', default='/unspecified_network', type=str, metavar='NET_NAME',
                    help='name of network ')     


parser.add_argument('--save_filter_file', default='model_accuracy', type=str, metavar='ACCURACY_FILENAME',
                    help='filename to save accuracy of model ')                      


args = parser.parse_args()

# def main_worker(gpu, ngpus_per_node, args):

global best_acc1

# create model
if args.pretrained:
    print("=> using pre-trained model '{}'".format(args.arch))
    model = models.__dict__[args.arch](pretrained=True)
else:
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

model = torch.nn.DataParallel(model).cuda()


optimizer = torch.optim.SGD(model.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)

# optionally resume from a checkpoint
if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))

        checkpoint = torch.load(args.resume)

        state_dict_new = OrderedDict()
            # for k, v in checkpoint.items():
        for k, v in checkpoint['state_dict'].items():
            name = k.replace(".module", '') # remove 'module.' of dataparallel
            state_dict_new[name]=v


        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']

  
        print(model) #NB Prints ResNet with Kernel Size 7x7!!!!! NOT out Epoch!

        model.load_state_dict(state_dict_new)

        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

cudnn.benchmark = True

model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


extractor = Extractor(list(model.modules()))


extractor.activate()

# # Visualising the filters
plt.figure(figsize=(35, 35))


# change CNN_weights[0] to CNN_weights[1] gives completly black fields...
# presumably this is the layer!
for index, filter in enumerate(extractor.CNN_weights[0]):

    # print(type(filter)) #<class 'torch.Tensor'>
    # print(filter.size()) #torch.Size([3, 21, 21])
    # # print(filter)
    # print(index)
    # print(filter)
    # print(type(filter))


    # Receptive field as a numpy arrray
    np_arr = filter.cpu().detach().numpy()
    # print(np_arr)
    
    # file = 'test/test_'+str(index)+'.npy'
    
    parent_outdir = '/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/visualizing_filters/conv1_filters'
    
    # DONE
    # supRN50_conv1_21_g0_60e_e60_np_arr
    # supRN50_conv1_21_g4_60e_e60_np_arr
    # supRN50_conv1_21_g0_30e_g4_30e_e60_np_arr
        # /data/blurry_vision_sup_RN50/supervised_resnet50_conv1_21_from_gauss_0_for_30_epoch_to_gauss_4_for_30/outmodel/checkpoint_supervised_resnet50_conv1_21_from_gauss_0_for_30_epoch_to_gauss_4_for_30_epoch60.pth.tar
    # supRN50_conv1_21_g0_30e_g4_30e_e35_np_arr
        # /data/blurry_vision_sup_RN50/supervised_resnet50_conv1_21_from_gauss_0_for_30_epoch_to_gauss_4_for_30/outmodel/checkpoint_supervised_resnet50_conv1_21_from_gauss_0_for_30_epoch_to_gauss_4_for_30_epoch35.pth.tar
    # supRN50_conv1_21_g4_30e_g0_30e_e35_np_arr
        # /data/blurry_vision_sup_RN50/supervised_resnet50_conv1_21_from_gauss_4_for_30_epoch_to_gauss_0_for_30/outmodel/checkpoint_supervised_resnet50_conv1_21_from_gauss_4_for_30_epoch_to_gauss_0_for_30_epoch35.pth.tar
    # supRN50_conv1_21_g4_30e_g0_30e_e60_np_arr
        # /data/blurry_vision_sup_RN50/supervised_resnet50_conv1_21_from_gauss_4_for_30_epoch_to_gauss_0_for_30/outmodel/checkpoint_supervised_resnet50_conv1_21_from_gauss_4_for_30_epoch_to_gauss_0_for_30_epoch60.pth.tar
    # supRN50_conv1_21_g0_60e_e35_np_arr
        # /data/blurry_vision_sup_RN50/supervised_resnet50_conv1_21_gauss_0_for_60_epoch/outmodel/checkpoint_supervised_resnet50_conv1_21_gauss_0_for_60_epoch_epoch35.pth.tar
    # supRN50_conv1_21_g4_60e_e35_np_arr
        # /data/blurry_vision_sup_RN50/supervised_resnet50_conv1_21_gauss_4_for_60_epoch/outmodel/checkpoint_supervised_resnet50_conv1_21_gauss_4_for_60_epoch_epoch35.pth.tar
    

    
    

  

    # for epoch in 
    # Just 2 for now! All eventually!
    # epochs_to_run = [35, 60]
    
    
    # CHANGE THIS AND PATH ONLY!
    filter_label = 'supRN50_conv1_21_g0_30e_g4_30e_e35'
    
    outdir = os.path.join(parent_outdir, f'{filter_label}_np_arr')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        print('Making directory!')
    else:
        print('Directory exists!')
    
    file = f'{outdir}/{filter_label}_filter_np_arr_'+str(index)+'.npy'
    numpy.save(file, np_arr, allow_pickle=True, fix_imports=True)
    print(f'Filter {index}:size - {np_arr.shape} (sanity check: are elenetts 1 and 2 both 21)') #(3, 21, 21)

#   CAUTION: RUN WITH THE CORRECT COMMANT OTHERWISE LOADS RESNET 18!
# (blurry_vision) ainedineen@cusacklab-lamb00:~/blurry_vision/pytorch_untrained_models/imagenet/visualizing_filters$ python Visualiser_rf_21_single_field_test_to_np.py 
# => creating model 'resnet18'


# python3 Visualiser_rf_21_single_field_test.py --a  'resnet50' --resume "/data/blurry_vision_sup_RN50/supervised_resnet50_conv1_21_gauss_0_for_60_epoch/outmodel/checkpoint_supervised_resnet50_conv1_21_gauss_0_for_60_epoch_epoch60.pth.tar"
# python3 Visualiser_rf_21_single_field_test.py --a  'resnet50' --resume "/data/blurry_vision_sup_RN50/supervised_resnet50_conv1_21_gauss_0_for_60_epoch/outmodel/checkpoint_supervised_resnet50_conv1_21_gauss_0_for_60_epoch_epoch60.pth.tar"
# python3 Visualiser_rf_21_single_field_test.py --a  'resnet50' --resume "/data/blurry_vision_sup_RN50/supervised_resnet50_conv1_21_from_gauss_0_for_30_epoch_to_gauss_4_for_30/outmodel/checkpoint_supervised_resnet50_conv1_21_from_gauss_0_for_30_epoch_to_gauss_4_for_30_epoch60.pth.tar"
# python3 Visualiser_rf_21_single_field_test_to_np.py --a  'resnet50' --resume "/data/blurry_vision_sup_RN50/supervised_resnet50_conv1_21_from_gauss_0_for_30_epoch_to_gauss_4_for_30/outmodel/checkpoint_supervised_resnet50_conv1_21_from_gauss_0_for_30_epoch_to_gauss_4_for_30_epoch35.pth.tar"