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
                    help='use pre-trained model')


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

# define loss function (criterion) and optimizer
# criterion = nn.CrossEntropyLoss().cuda(NONE) #Do not need?

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
        # model.load_state_dict(checkpoint['state_dict'])

        model.load_state_dict(state_dict_new)

        # model = model.load_state_dict(checkpoint['state_dict']) #TRY - still printing untrained rfs
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

cudnn.benchmark = True

model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loading the model
# resnet = models.resnet50()
# extractor = Extractor(list(resnet.children()))
extractor = Extractor(list(model.children()))

extractor.activate()


# Visualising the filters
plt.figure(figsize=(35, 35))
for index, filter in enumerate(extractor.CNN_weights[0]):
    plt.subplot(8, 8, index + 1)
    plt.imshow(filter[0, :, :].detach(), cmap='gray')
    plt.axis('off')

plt.show()
plt.savefig('/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/visualizing_filters/test_out2.png')
    # plt.show()


# Filter Map
img = cv.cvtColor(cv.imread('Featuremaps&Filters/img.png'), cv.COLOR_BGR2RGB)
img = t.Compose([
    t.ToPILImage(),
    t.Resize((128, 128)),
    # t.Grayscale(),
    t.ToTensor(),
    t.Normalize(0.5, 0.5)])(img).unsqueeze(0)

featuremaps = [extractor.CNN_layers[0](img)]
for x in range(1, len(extractor.CNN_layers)):
    featuremaps.append(extractor.CNN_layers[x](featuremaps[-1]))

# Visualising the featuremaps
for x in range(len(featuremaps)):
    plt.figure(figsize=(30, 30))
    layers = featuremaps[x][0, :, :, :].detach()
    for i, filter in enumerate(layers):
        if i == 64:
            break
        plt.subplot(8, 8, i + 1)
        plt.imshow(filter, cmap='gray')
        plt.axis('off')

    # plt.savefig('featuremap%s.png'%(x))

plt.show()


