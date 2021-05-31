# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models


class ImageNet_Norm_Layer_2(nn.Module):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super(ImageNet_Norm_Layer_2, self).__init__()
        self.cuda = torch.cuda.is_available()
        dtype = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        self.mean = Variable(torch.FloatTensor(mean).type(dtype), requires_grad=0)
        self.std = Variable(torch.FloatTensor(std).type(dtype), requires_grad=0)

    def forward(self, input):
        return ((input.permute(0, 2, 3, 1) - self.mean) / self.std).permute(0, 3, 1, 2)

class regressor(nn.Module):
    def __init__(self, inplanes):
        super(regressor, self).__init__()
        self.fc = nn.Linear(inplanes,1024)
        self.avgpooling = nn.AdaptiveAvgPool2d(1)



        self.fc_q = nn.Linear(1024, 4, bias=False)
        self.fc_t = nn.Linear(1024, 3, bias=False)

    def forward(self,x):
        x = self.avgpooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        q = self.fc_q(x)
        t = self.fc_t(x)
        return x, q, t


class Res50PoseRess(nn.Module):
    def __init__(self):
        super(Res50PoseRess, self).__init__()

        self.norm_layer = ImageNet_Norm_Layer_2()

        Res50 = models.resnet50(pretrained=True)
        # print(Res50)
        self.conv1 = nn.Sequential(Res50.conv1,Res50.bn1,Res50.relu,Res50.maxpool)
        self.layer1 = Res50.layer1
        self.layer2 = Res50.layer2
        self.layer3 = Res50.layer3
        self.layer4 = Res50.layer4
        self.pred_1 = self._make_pre_fc(regressor,256)
        self.pred_2 = self._make_pre_fc(regressor, 512)
        self.pred_3 = self._make_pre_fc(regressor, 1024)
        self.pred_4 = self._make_pre_fc(regressor, 2048)




    def _make_pre_fc(self,block,inplanes):

        return block(inplanes)



    def forward(self, x_base):
        x_base = self.norm_layer(x_base)
        x_conv1 = self.conv1(x_base)
        x_layer1 = self.layer1(x_conv1)
        x_layer2 = self.layer2(x_layer1)
        x_layer3 = self.layer3(x_layer2)
        x_layer4 = self.layer4(x_layer3)

        x_1,x_1_q,x_1_t = self.pred_1(x_layer1)
        x_2, x_2_q, x_2_t = self.pred_2(x_layer2)
        x_3, x_3_q, x_3_t = self.pred_3(x_layer3)
        x_4, x_4_q, x_4_t = self.pred_4(x_layer4)


        return x_layer1,x_layer2,x_layer3,x_layer4,x_1_q,x_1_t,x_2_q,x_2_t, x_3_q,x_3_t,x_4_q,x_4_t




class regressor_2(nn.Module):
    def __init__(self, inplanes):
        super(regressor_2, self).__init__()
        self.fc = nn.Linear(inplanes,1024)
        self.avgpooling = nn.AdaptiveAvgPool2d(1)



        self.fc_q = nn.Linear(inplanes, 4, bias=False)
        self.fc_t = nn.Linear(inplanes, 3, bias=False)

    def forward(self,x):
        x = self.avgpooling(x)
        x = x.view(x.size(0), -1)
        # x = self.fc(x)

        q = self.fc_q(x)
        t = self.fc_t(x)
        return x, q, t


class Res50PoseRess2(nn.Module):
    def __init__(self):
        super(Res50PoseRess2, self).__init__()

        self.norm_layer = ImageNet_Norm_Layer_2()

        Res50 = models.resnet50(pretrained=True)
        print(Res50)
        self.conv1 = nn.Sequential(Res50.conv1,Res50.bn1,Res50.relu,Res50.maxpool)
        self.layer1 = Res50.layer1
        self.layer2 = Res50.layer2
        self.layer3 = Res50.layer3
        self.layer4 = Res50.layer4
        self.pred_1 = self._make_pre_fc(regressor_2,256)
        self.pred_2 = self._make_pre_fc(regressor_2, 512)
        self.pred_3 = self._make_pre_fc(regressor_2, 1024)
        self.pred_4 = self._make_pre_fc(regressor_2, 2048)




    def _make_pre_fc(self,block,inplanes):

        return block(inplanes)



    def forward(self, x_base):
        x_base = self.norm_layer(x_base)
        x_conv1 = self.conv1(x_base)
        x_layer1 = self.layer1(x_conv1)
        x_layer2 = self.layer2(x_layer1)
        x_layer3 = self.layer3(x_layer2)
        x_layer4 = self.layer4(x_layer3)

        x_1,x_1_q,x_1_t = self.pred_1(x_layer1)
        x_2, x_2_q, x_2_t = self.pred_2(x_layer2)
        x_3, x_3_q, x_3_t = self.pred_3(x_layer3)
        x_4, x_4_q, x_4_t = self.pred_4(x_layer4)


        return x_1,x_2,x_3,x_4,x_1_q,x_1_t,x_2_q,x_2_t, x_3_q,x_3_t,x_4_q,x_4_t



class Res50PoseRess3(nn.Module):
    def __init__(self):
        super(Res50PoseRess3, self).__init__()

        self.norm_layer = ImageNet_Norm_Layer_2()

        Res50 = models.resnet50(pretrained=True)
        print(Res50)
        self.conv1 = nn.Sequential(Res50.conv1,Res50.bn1,Res50.relu,Res50.maxpool)
        self.layer1 = Res50.layer1
        self.layer2 = Res50.layer2
        self.layer3 = Res50.layer3
        self.layer4 = Res50.layer4
        self.avgpooling = nn.AdaptiveAvgPool2d(1)
        self.fc_q = nn.Linear(3840,4)
        self.fc_t = nn.Linear(3840, 3)


    def forward(self, x_base):
        x_base = self.norm_layer(x_base)
        x_conv1 = self.conv1(x_base)
        x_layer1 = self.layer1(x_conv1)
        x_layer2 = self.layer2(x_layer1)
        x_layer3 = self.layer3(x_layer2)
        x_layer4 = self.layer4(x_layer3)



        x_1 = self.avgpooling(x_layer1)
        x_2 = self.avgpooling(x_layer2)
        x_3 = self.avgpooling(x_layer3)
        x_4= self.avgpooling(x_layer4)
        x = torch.cat((x_1,x_2,x_3,x_4),1)
        x = x.view(x.size(0), -1)
        x_q = self.fc_q(x)
        x_t = self.fc_t(x)



        return x_t, x_q,x
