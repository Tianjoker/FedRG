import copy

import torch
import torch.nn.functional as F
import torchvision.models as models
from torch import nn



OneLayer = "1_layer"
TwoLayer = "2_layer"


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
    ):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def byol_loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size=4096, num_layer=TwoLayer):
        super().__init__()
        self.in_features = dim
        if num_layer == OneLayer:
            self.net = nn.Sequential(
                nn.Linear(dim, projection_size),
            )
        elif num_layer == TwoLayer:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, projection_size),
            )
        else:
            raise NotImplementedError(f"Not defined MLP: {num_layer}")

    def forward(self, x):
        return self.net(x)

class BYOLModel(nn.Module):
    def __init__(self,
            net,
            num_classes=10,
            projection_size=2048,
            projection_hidden_size=4096,
            moving_average_decay=0.99,
            stop_gradient=True,
            has_predictor=True,
            predictor_network=TwoLayer):
        super(BYOLModel, self).__init__()

        self.online_encoder = net
        feature_dim = net.feature_dim
        self.online_encoder.fc = MLP(feature_dim, projection_size, projection_hidden_size)  # projector

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size, predictor_network)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_ema_updater = EMA(moving_average_decay)

        self.stop_gradient = stop_gradient
        self.has_predictor = has_predictor
        self.class_predictor = nn.Sequential(nn.Linear(512, 256), nn.Linear(256, num_classes))


    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert (
                self.target_encoder is not None
        ), "target encoder has not been created yet"
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)


    def forward(self, image_one, image_two):
        online_f_one, online_pred_one = self.online_encoder(image_one)
        online_f_two, online_pred_two = self.online_encoder(image_two)


        if self.has_predictor:
            online_pred_one = self.online_predictor(online_pred_one)
            online_pred_two = self.online_predictor(online_pred_two)

        if self.stop_gradient:
            with torch.no_grad():
                _, target_proj_one = self.target_encoder(image_one)
                _, target_proj_two = self.target_encoder(image_two)

                target_proj_one = target_proj_one.detach()
                target_proj_two = target_proj_two.detach()

        else:
            if self.target_encoder is None:
                self.target_encoder = self._get_target_encoder()
            _, target_proj_one = self.target_encoder(image_one)
            _, target_proj_two = self.target_encoder(image_two)


        loss_one = byol_loss_fn(online_pred_one, target_proj_two)
        loss_two = byol_loss_fn(online_pred_two, target_proj_one)
        loss = loss_one + loss_two
        
        return online_f_one, online_f_two, loss.mean()


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class SimCLRModel(nn.Module):
    def __init__(self, net,
                 projection_size=2048,
                 projection_hidden_size=4096):
        super().__init__()
        self.encoder = nn.Sequential(
            net.conv1, net.bn1, nn.ReLU(),
            net.layer1, net.layer2, net.layer3, net.layer4, net.avgpool
        )
        self.feature_dim = net.feature_dim  # 512 for ResNet18
        self.project_dim = projection_size # 2048 for projector
        self.projector = MLP(self.feature_dim,
                             projection_size,
                             projection_hidden_size)

    def forward(self, x):
        out = self.encoder(x)                # [B, 512, 1, 1]
        feat = torch.flatten(out, 1)         # [B, 512]
        proj = self.projector(feat)          # [B, 2048]
        return feat, proj

    def encoder_forward(self, x, l2=False):
        out = self.encoder(x)
        h = torch.flatten(out, 1)  # [B, 512]
        if l2:
            h = h / (h.norm(dim=1, keepdim=True) + 1e-8)
        return h

def D(p, z):  # negative cosine similarity
    return - F.cosine_similarity(p, z.detach(), dim=-1).mean()


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=512):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.num_layers = 3

    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x

class prediction_MLP(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=256, out_dim=512):  # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class SimSiamModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.encoder = backbone
        self.encoder.fc = projection_MLP(512)

        self.predictor = prediction_MLP()

    def forward(self, x1, x2):
        f, h = self.encoder, self.predictor
        _, z1 = f(x1)
        _, z2 =  f(x2)
        p1, p2 = h(z1), h(z2)
        loss = D(p1, z2) / 2 + D(p2, z1) / 2
        p = torch.cat((p1,p2), dim=0)
        z = torch.cat((z1, z2), dim=0)
        return loss, z

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class DINO(nn.Module):
    def __init__(self, net, feature_dim=512, hidden_dim=2048,
                 out_dim=2048, momentum=0.99):
        super().__init__()

        # ===== Student =====
        self.student_encoder = copy.deepcopy(net)
        self.student_encoder.linear = nn.Identity()
        self.student_projector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

        # ===== Teacher =====
        self.teacher_encoder = copy.deepcopy(net)
        self.teacher_encoder.linear = nn.Identity()
        self.teacher_projector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

        # 初始化 teacher = student
        self.teacher_encoder.load_state_dict(self.student_encoder.state_dict())
        self.teacher_projector.load_state_dict(self.student_projector.state_dict())

        # 冻结 teacher 参数
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False
        for p in self.teacher_projector.parameters():
            p.requires_grad = False


        for m in self.teacher_encoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.track_running_stats = False

        self.momentum = momentum

    def forward_student(self, x):
        feat, _ = self.student_encoder(x)       # [B, feature_dim], _
        proj = self.student_projector(feat)     # [B, out_dim]
        return feat, proj

    def forward_teacher(self, x):
        with torch.no_grad():
            feat, _ = self.teacher_encoder(x)
            proj = self.teacher_projector(feat)
        return feat, proj

    @torch.no_grad()
    def update_teacher(self):
        # EMA 更新
        for param_t, param_s in zip(self.teacher_encoder.parameters(),
                                    self.student_encoder.parameters()):
            param_t.data.mul_(self.momentum).add_(param_s.data, alpha=1 - self.momentum)
        for param_t, param_s in zip(self.teacher_projector.parameters(),
                                    self.student_projector.parameters()):
            param_t.data.mul_(self.momentum).add_(param_s.data, alpha=1 - self.momentum)
