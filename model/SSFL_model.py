import torch
import torch.nn as nn
import torch.nn.functional as F

from model.MLP import MLP

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class GlobalModel(nn.Module):
    def __init__(self, 
                 args=None, 
                 num_classes=10, 
                 input_channels=3,
                 projection_size=2048,
                 projection_hidden_size=4086):
        super(GlobalModel, self).__init__()
        self.in_planes = 64

        self.block = BasicBlock
        self.num_blocks = [2,2,2,2]
        self.args = args

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(self.block, 64, self.num_blocks[0], stride=1)
        self.layer2 = self._make_layer(self.block, 128, self.num_blocks[1], stride=2)
        self.layer3 = self._make_layer(self.block, 256, self.num_blocks[2], stride=2)
        self.layer4 = self._make_layer(self.block, 512, self.num_blocks[3], stride=2)
        if self.args.su_unsup_decouple:
            self.unsup_linear = MLP(512*self.block.expansion, projection_size, projection_hidden_size)
            self.sup_linear = nn.Linear(512*self.block.expansion, num_classes)
        else:
            self.unsup_linear = MLP(512*self.block.expansion, projection_size, projection_hidden_size)
            self.sup_linear = nn.Linear(projection_size, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.layers_name_map = {
            "classifier": "linear"
        }

        inplanes = [64, 64, 128, 256, 512]
        inplanes = [ inplane * self.block.expansion for inplane in inplanes]
        # logging.info(inplanes)


    def _make_layer(self, block, planes, num_blocks, stride):
        
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, images1, images2):
        images = torch.cat((images1, images2), dim=0)
        out = F.relu(self.bn1(self.conv1(images)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        feat = out.view(out.size(0), -1) * 1.0
        if self.args.su_unsup_decouple:
            unsup_logits = self.unsup_linear(feat)
            sup_logits = self.sup_linear(feat)
            return feat, unsup_logits, sup_logits
        else:
            unsup_logits = self.unsup_linear(feat)
            sup_logits = self.sup_linear(unsup_logits)
            return feat, unsup_logits, sup_logits
        
    def inference(self, images1):
        images = images1
        out = F.relu(self.bn1(self.conv1(images)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        feat = out.view(out.size(0), -1) * 1.0
        if self.args.su_unsup_decouple:
            unsup_logits = self.unsup_linear(feat)
            sup_logits = self.sup_linear(feat)
            return sup_logits
        else:
            unsup_logits = self.unsup_linear(feat)
            sup_logits = self.sup_linear(unsup_logits)
            return  sup_logits
        
class LocalUnsupModel(nn.Module):
    def __init__(self, 
                 args=None,
                 input_channels=3,
                 projection_size=2048,
                 projection_hidden_size=4086):
        super(LocalUnsupModel, self).__init__()
        self.in_planes = 64

        self.block = BasicBlock
        self.num_blocks = [2,2,2,2]
        self.args = args

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(self.block, 64, self.num_blocks[0], stride=1)
        self.layer2 = self._make_layer(self.block, 128, self.num_blocks[1], stride=2)
        self.layer3 = self._make_layer(self.block, 256, self.num_blocks[2], stride=2)
        self.layer4 = self._make_layer(self.block, 512, self.num_blocks[3], stride=2)
        self.fc = MLP(512*self.block.expansion, projection_size, projection_hidden_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.layers_name_map = {
            "classifier": "linear"
        }

        inplanes = [64, 64, 128, 256, 512]
        inplanes = [ inplane * self.block.expansion for inplane in inplanes]
        # logging.info(inplanes)


    def _make_layer(self, block, planes, num_blocks, stride):
        
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, images1, images2):
        images = torch.cat((images1, images2), dim=0)
        out = F.relu(self.bn1(self.conv1(images)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        feat = out.view(out.size(0), -1) * 1.0
        out = self.fc(feat)
        return feat, out


class GlobalModel_CCL(nn.Module):
    def __init__(self, 
                 args=None, 
                 num_classes=10, 
                 input_channels=3,
                 projection_size=2048,
                 projection_hidden_size=4086):
        super(GlobalModel_CCL, self).__init__()
        self.in_planes = 64

        self.block = BasicBlock
        self.num_blocks = [2,2,2,2]
        self.args = args

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(self.block, 64, self.num_blocks[0], stride=1)
        self.layer2 = self._make_layer(self.block, 128, self.num_blocks[1], stride=2)
        self.layer3 = self._make_layer(self.block, 256, self.num_blocks[2], stride=2)
        self.layer4 = self._make_layer(self.block, 512, self.num_blocks[3], stride=2)
        self.unsup_linear = MLP(512*self.block.expansion, projection_size, projection_hidden_size)
        self.sup_linear = nn.Linear(projection_size, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.layers_name_map = {
            "classifier": "linear"
        }

        inplanes = [64, 64, 128, 256, 512]
        inplanes = [ inplane * self.block.expansion for inplane in inplanes]
        # logging.info(inplanes)


    def _make_layer(self, block, planes, num_blocks, stride):
        
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def f(self, images):
        out = F.relu(self.bn1(self.conv1(images)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)        
        out = self.avgpool(out)
        feat = out.view(out.size(0), -1) * 1.0
        unsup_logits = self.unsup_linear(feat)
        sup_logits = self.sup_linear(unsup_logits)
        return feat, unsup_logits, sup_logits
    
    def forward(self, images1, images2, perturb=None, cls_adv_per=None, instance_adv_per=None):
        images = torch.cat((images1, images2), dim=0)
        if perturb is not None:
            images += perturb
        
        if cls_adv_per is not None and instance_adv_per is not None:
            images_instance_adv_per = images+instance_adv_per
            _,unsup_adv_logits,_ = self.f(images_instance_adv_per)
            images_cls_adv_per = images + cls_adv_per
            _, _, sup_adv_logits = self.f(images_cls_adv_per)
        feat, unsup_logits, sup_logits = self.f(images)
        if cls_adv_per is not None and instance_adv_per is not None:
            return feat, unsup_logits, sup_logits, unsup_adv_logits, sup_adv_logits
        else:
            return feat, unsup_logits, sup_logits 

        out = F.relu(self.bn1(self.conv1(images)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        feat = out.view(out.size(0), -1) * 1.0
        
        if perturb is not None:
            feat += perturb
        if cls_adv_per is not None and instance_adv_per is not None:
            feat_instance_adv_per = feat+instance_adv_per
            unsup_adv_logits = self.unsup_linear(feat_instance_adv_per)
            feat_cls_adv_per = feat + cls_adv_per
            sup_adv_logits = self.sup_linear(self.unsup_linear(feat_cls_adv_per))
        unsup_logits = self.unsup_linear(feat)
        sup_logits = self.sup_linear(unsup_logits)
        if cls_adv_per is not None and instance_adv_per is not None:
            return feat, unsup_logits, sup_logits, unsup_adv_logits, sup_adv_logits
        else:
            return feat, unsup_logits, sup_logits
        
    def inference(self, images1):
        images = images1
        out = F.relu(self.bn1(self.conv1(images)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        feat = out.view(out.size(0), -1) * 1.0
        unsup_logits = self.unsup_linear(feat)
        sup_logits = self.sup_linear(unsup_logits)
        return  sup_logits