import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0, 1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def conv_bn_relu(in_channels, out_channels, kernel_size, \
        stride=1, padding=0, bn=True, relu=True):
    bias = not bn
    layers = []
    layers.append(
        nn.Conv2d(in_channels,
                  out_channels,
                  kernel_size,
                  stride,
                  padding,
                  bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers = nn.Sequential(*layers)

    # initialize the weights
    for m in layers.modules():
        init_weights(m)

    return layers

def convt_bn_relu(in_channels, out_channels, kernel_size, \
        stride=1, padding=0, output_padding=0, bn=True, relu=True):
    bias = not bn
    layers = []
    layers.append(
        nn.ConvTranspose2d(in_channels,
                           out_channels,
                           kernel_size,
                           stride,
                           padding,
                           output_padding,
                           bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers = nn.Sequential(*layers)

    # initialize the weights
    for m in layers.modules():
        init_weights(m)

    return layers


class TactileCompletionNet(nn.Module):
    def __init__(self, args):
        assert (
            args.layers in [18, 34, 50, 101, 152]
        ), 'Only layers 18, 34, 50, 101, and 152 are defined, but got {}'.format(
            layers)
        super(TactileCompletionNet, self).__init__()
        channels = 32
        
        if args.modality == 'g':
            self.conv1_t = conv_bn_relu(1,
                                        channels // 2,
                                        kernel_size=6,
                                        stride=2,
                                        padding=1)            
        else:
            self.conv1_t = conv_bn_relu(3,
                                        channels // 2,
                                        kernel_size=6,
                                        stride=2,
                                        padding=1)
            
        self.conv2_t = conv_bn_relu(channels // 2,
                                    channels,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1)
        
        self.conv3_t = conv_bn_relu(channels,
                                    channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
        

        self.conv1_v = conv_bn_relu(3,
                                    channels // 2,
                                    kernel_size=6,
                                    stride=2,
                                    padding=1)
            
        self.conv2_v = conv_bn_relu(channels // 2,
                                    channels,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1)
        self.conv3_v = conv_bn_relu(channels,
                                    channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)


        pretrained_model = resnet.__dict__['resnet{}'.format(
            args.layers)](pretrained=args.pretrained)
        if not args.pretrained:
            pretrained_model.apply(init_weights)
        #self.maxpool = pretrained_model._modules['maxpool']
        self.conv2 = pretrained_model._modules['layer1']
        self.conv3 = pretrained_model._modules['layer2']
        self.conv4 = pretrained_model._modules['layer3']
        self.conv5 = pretrained_model._modules['layer4']
        del pretrained_model  # clear memory

        # define number of intermediate channels
        if args.layers <= 34:
            num_channels = 512
        elif args.layers >= 50:
            num_channels = 2048
        self.conv6 = conv_bn_relu(num_channels, 
                                  512,
                                  kernel_size=3,
                                  stride=2,
                                  padding=1)

        # decoding layers
        kernel_size = 3
        stride = 2
        self.convt5 = convt_bn_relu(in_channels=512,
                                    out_channels=256,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=1,
                                    output_padding=1)
        self.convt4 = convt_bn_relu(in_channels=768,
                                    out_channels=128,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=1,
                                    output_padding=1)
        self.convt3 = convt_bn_relu(in_channels=(256 + 128),
                                    out_channels=64,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=1,
                                    output_padding=1)
        self.convt2 = convt_bn_relu(in_channels=(128 + 64),
                                    out_channels=64,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=1,
                                    output_padding=1)
        self.convt1 = convt_bn_relu(in_channels=128,
                                    out_channels=64,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=1)
        
        self.convtf_post_1 = convt_bn_relu(in_channels=128,
                                            out_channels=64,
                                            kernel_size=3,
                                            stride=1,
                                            padding = 1)
        self.convtf_post_2 = convt_bn_relu(in_channels=64,
                                            out_channels=32,
                                            kernel_size=3,
                                            stride=2,
                                            padding = 1)
        
        if args.modality == "g":
            self.convtf_post_3 = convt_bn_relu(in_channels=32,
                                                out_channels=1,
                                                kernel_size=6,
                                                stride=2,
                                                padding = 1)
        else:
            self.convtf_post_3 = convt_bn_relu(in_channels=32,
                                                out_channels=3,
                                                kernel_size=6,
                                                stride=2,
                                                padding = 1)

    def forward(self, x):
        # first layer
        conv1_t = self.conv1_t(x['t'])
        conv2_t = self.conv2_t(conv1_t)
        conv3_t = self.conv3_t(conv2_t)
        
        conv1_v = self.conv1_v(x['v'])
        conv2_v = self.conv2_v(conv1_v)
        conv3_v = self.conv3_v(conv2_v)
        
        conv1 = torch.cat((conv3_v, conv3_t), 1)
        
        # # peform additive fusion
        # conv1 = conv1_t + conv1_v

        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)  # batchsize * ? * 128 * 128
        conv4 = self.conv4(conv3)  # batchsize * ? * 64 * 64
        conv5 = self.conv5(conv4)  # batchsize * ? * 32 * 32
        conv6 = self.conv6(conv5)  # batchsize * ? * 16 * 16
        

        # decoder
        convt5 = self.convt5(conv6)
        
        
        # conv5 = F.pad(conv5, (0, 1, 0, 1), "constant", 0)
        y = torch.cat((convt5, conv5), 1)

 
        convt4 = self.convt4(y)
        # conv4 = F.pad(conv4, (0, 2, 0, 2), "constant", 0)
        y = torch.cat((convt4, conv4), 1)

        convt3 = self.convt3(y)
        # conv3 = F.pad(conv3, (0, 4, 0, 4), "constant", 0)
        y = torch.cat((convt3, conv3), 1)


        convt2 = self.convt2(y)
        # conv2 = F.pad(conv2, (0, 8, 0, 9), "constant", 0)
        y = torch.cat((convt2, conv2), 1)

        convt1 = self.convt1(y)
        # conv1 = F.pad(conv1, (0, 8, 0, 9), "constant", 0)
        y = torch.cat((convt1, conv1), 1)
        
        y = self.convtf_post_1(y)
        
        y = self.convtf_post_2(y)
        
        y = self.convtf_post_3(y)
        
        # y = self.convtf_post_4(y)        
        
        y_upsampled = F.interpolate(y, size=(256, 256), mode='bilinear', align_corners=False) # # batchsize * 3 * 256 * 256
        
        y_output = torch.sigmoid(y_upsampled)
        return y_output
