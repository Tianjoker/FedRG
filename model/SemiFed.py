import torch
from torch import nn

from model.MLP import MLP

class SemiFed(nn.Module):
    def __init__(self, unsup_model,
                 semi_model,
                num_class=10, input_channel=3, use_projector=False):
        super().__init__()
        self.unsup_model = unsup_model         # SimCLR encoder

        self.semi_model = semi_model
        self.use_projector = use_projector

        in_dim = (unsup_model.project_dim if use_projector else unsup_model.feature_dim)

        self.classifier_head = nn.Linear(in_dim, num_class)
        self.T = nn.Parameter(torch.eye(num_class))

        self.output_dim  = num_class
        self.num_class = num_class
        self.input_channel = input_channel

    def set_projector_trainable(self, flag: bool):
        for p in self.unsup_model.projector.parameters():
            p.requires_grad = flag

        self.unsup_model.projector.train(flag)

    def forward(self, image1, image2):
        images = torch.cat((image1, image2), dim=0)


        # else:
        if self.semi_model == 'SimCLR':

            feat = self.unsup_model.encoder_forward(images)  # [B, 512]

            if self.use_projector:

                proj = self.unsup_model.projector(feat)  # [B, 2048]
            else:
                proj = None


            return feat, proj
        return

    def inference(self, x):

        if self.use_projector:
            _, feats = self.unsup_model(x)  # projector 输出
        else:
            feats = self.unsup_model.encoder_forward(x)  # ★ encoder 输出
        logits = self.classifier_head(feats)
        return feats, logits