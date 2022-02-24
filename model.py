import os
import torch
import torchvision

import clip
from torchvision.models.detection.roi_heads import fastrcnn_loss, keypointrcnn_inference, keypointrcnn_loss

import config
from torch import nn
from PIL import Image
from torch import Tensor

from utils import FeatureExtractor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.cuda.amp import autocast

def create_model(model_type='Vanilla', classes=[]):
    if model_type == 'Vanilla':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    if model_type == 'Custom-Vanilla':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, (len(classes)))

    if model_type == 'Fully-Custom-Vanilla':

        # load a pre-trained model for classification and return
        # only the features
        backbone = FeatureExtractor(torchvision.models.resnet50(pretrained=True),
                                    layers=['layer4.2'])
        # FasterRCNN needs to know the number of
        # output channels in a backbone. For Faster-RCNN, it's 256
        # so we need to add it here
        backbone.out_channels = 2048

        # freeze the parameters in the backbone
        for child in backbone.children():
            for param in child.parameters():
                param.requires_grad = False

        # let's make the RPN generate 5 x 3 anchors per spatial
        # location, with 5 different sizes and 3 different aspect
        # ratios. We have a Tuple[Tuple[int]] because each feature
        # map could potentially have different sizes and
        # aspect ratios
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                           aspect_ratios=((0.5, 1.0, 2.0),))

        # let's define what are the feature maps that we will
        # use to perform the region of interest cropping, as well as
        # the size of the crop after rescaling.
        # if your backbone returns a Tensor, featmap_names is expected to
        # be [0]. More generally, the backbone should return an
        # OrderedDict[Tensor], and in featmap_names you can choose which
        # feature maps to use.
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                        output_size=7,
                                                        sampling_ratio=2)

        # put the pieces together inside a FasterRCNN model
        model = FasterRCNN(backbone,
                           num_classes=len(classes),
                           rpn_anchor_generator=anchor_generator,
                           box_roi_pool=roi_pooler)

    if model_type == 'CLIP-Backbone-FRCNN' or model_type == 'CLIP-FRCNN':
        # Load clip backbone and feature extractor
        model, preprocess = clip.load("RN50", device=config.DEVICE)
        model.eval()
        model.float()

        del model.visual.attnpool  # delete the attention pool, so that we can feed the model larger images
        model.visual.attnpool = nn.Identity()  # add an Identity layer so that we can use the model, else it complains

        backbone = FeatureExtractor(model.visual,
                                    layers=['layer4.2']).eval()  # use forward hooks to grab the feature extractor

        # freeze the parameters in the backbone
        for child in backbone.children():
            for param in child.parameters():
                param.requires_grad = False

        # FasterRCNN needs to know the number of output channels in a backbone.
        backbone.out_channels = \
        backbone(torch.rand(1, 3, config.INPUT_RESOLUTION, config.INPUT_RESOLUTION).to(config.DEVICE)).shape[1]

        # let's make the RPN generate 5 x 3 anchors per spatial
        # location, with 5 different sizes and 3 different aspect
        # ratios. We have a Tuple[Tuple[int]] because each feature
        # map could potentially have different sizes and
        # aspect ratios
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                           aspect_ratios=((0.5, 1.0, 2.0),))

        # let's define what are the feature maps that we will
        # use to perform the region of interest cropping, as well as
        # the size of the crop after rescaling.
        # if your backbone returns a Tensor, featmap_names is expected to
        # be [0]. More generally, the backbone should return an
        # OrderedDict[Tensor], and in featmap_names you can choose which
        # feature maps to use.
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                        output_size=7,
                                                        sampling_ratio=8)

        model = FasterRCNN(backbone,
                           num_classes=len(classes),
                           rpn_anchor_generator=anchor_generator,
                           box_roi_pool=roi_pooler,
                           image_mean=config.MEAN,
                           image_std=config.STD,
                           ).to(config.DEVICE)

        if model_type == 'CLIP-FRCNN':
            with autocast():
                model.roi_heads.box_head = CLIPHead()
                model.roi_heads.box_predictor = CLIPRCNNPredictor(1024, classes)  # CLIP embeds into 1024 dimensions for the RN50 implementation

            #we do not want to train the predictor since it is embedding into CLIP space
            for child in model.roi_heads.box_head.children():
                for p in child.parameters():
                    p.requires_grad = False
        model.float()
    return model


class test_backbone():
    def __init__(self, ):
        self.CLIP_model, preprocess = clip.load("RN50", device=config.DEVICE)
        self.CLIP_model.eval()
        self.CLIP_model.float()

        self.test_image = preprocess(Image.open(os.path.join('test.jpg')).convert("RGB")).unsqueeze(0).to(config.DEVICE)
        self.image_embedder = list(self.CLIP_model.visual.children())[-1].float().cuda().eval().float()  # take the last layer manually
        self.CLIP_image_features = self.CLIP_model.encode_image(self.test_image)

    def test(self, model):
        model.eval()
        model_image_features = self.image_embedder(model.backbone(self.test_image))
        assert (torch.all(model_image_features.eq(self.CLIP_image_features)))


class CLIPHead(nn.Module):
    """
    heads for FPN-based models

"""

    def __init__(self, ):
        super(CLIPHead, self).__init__()
        CLIP_model, _ = clip.load("RN50", device=config.DEVICE)
        self.image_embedder = list(CLIP_model.visual.children())[-1].cuda().eval()
        del CLIP_model

    def forward(self, x):
        x = self.image_embedder(x)
        x /= x.norm(dim=-1, keepdim=True)
        return x


class CLIPRCNNPredictor(nn.Module):
    """
    CLIP classification + bounding box regression layers
    for Fast R-CNN.

    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, text):
        super(CLIPRCNNPredictor, self).__init__()
        CLIP_model, _ = clip.load("RN50", device=config.DEVICE)
        CLIP_model.eval()
        CLIP_model.float()

        self.text_features = CLIP_model.encode_text(text).to(config.DEVICE).float()
        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

        self.bbox_pred = nn.Linear(in_channels, (len(text)*4)).to(config.DEVICE)
        del CLIP_model

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        # TODO: scores should be the logits, roi_heads takes the logits and performs the softmax.  It also drops all of the first row (assumed to be background)
        scores = (100.0 * x @ self.text_features.T)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas
