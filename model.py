import os
import torch
import torchvision

import clip
from torchvision.models.detection.roi_heads import fastrcnn_loss

import config
from torch import nn
from PIL import Image
from torch import Tensor

from utils import FeatureExtractor

from torchvision.models.detection.rpn import AnchorGenerator


from torchvision.models.detection import FasterRCNN

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

        #see args in faster rcnn
        model = FasterRCNN(backbone,
                           num_classes=len(classes),
                           rpn_anchor_generator=anchor_generator,
                           box_roi_pool=roi_pooler,
                           image_mean=config.MEAN,
                           image_std=config.STD,
                           rpn_fg_iou_thresh = .95,
                           rpn_bg_iou_thresh = .45,
                           rpn_nms_thresh = .7,
                           rpn_score_thresh = .9,

                           ).to(config.DEVICE)

        if model_type == 'CLIP-FRCNN':
            with autocast():

                model.roi_heads.box_head = CLIPHead()
                model.roi_heads.box_predictor = CLIPRCNNPredictor((2048*7*7), classes)  # CLIP space is 2048*7*7 for the RN50 implementation


            #we do not want to train the predictor since it is embedding into CLIP space
            for child in model.roi_heads.box_head.children():
                for p in child.parameters():
                    p.requires_grad = False



            # replace the loss function with one that ignores classification loss
            torchvision.models.detection.roi_heads.fastrcnn_loss = cliprcnn_loss
        model.float()

    if model_type == "CLIP-RPN":
        from torchvision.models.detection.transform import GeneralizedRCNNTransform
        from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork
        from torchvision.ops import MultiScaleRoIAlign

        from custom_rpn import ZeroShotOD

        model, _ = clip.load("RN50", device=config.DEVICE)
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

        del model
        # let's make the RPN generate 5 x 3 anchors per spatial
        # location, with 5 different sizes and 3 different aspect
        # ratios. We have a Tuple[Tuple[int]] because each feature
        # map could potentially have different sizes and
        # aspect ratios
        rpn_anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
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
        out_channels = backbone.out_channels

        rpn_head = RPNHead(
            out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
        )

        rpn_pre_nms_top_n_train = 500
        rpn_pre_nms_top_n_test = 500
        rpn_post_nms_top_n_train = 200
        rpn_post_nms_top_n_test = 200
        rpn_nms_thresh = 0.7
        rpn_fg_iou_thresh = 0.7
        rpn_bg_iou_thresh = 0.3
        rpn_batch_size_per_image = 256
        rpn_positive_fraction = 0.5
        rpn_score_thresh = 0.0

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
            score_thresh=rpn_score_thresh)

        min_size = 800
        max_size = 1333
        image_mean = config.MEAN,
        image_std = config.STD,

        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean[0], image_std[0])

        classifier = CLIPRPNPredictor((2048 * 7 * 7), classes)

        model = ZeroShotOD(backbone, rpn, roi_pooler, classifier, transform)

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
    This class gets the features from the pooled Regions of Interest, for the custom method using CLIP, we simply pass the roi on through
    heads for FPN-based models
"""

    def __init__(self, ):
        super(CLIPHead, self).__init__()
        # CLIP_model, _ = clip.load("RN50", device=config.DEVICE)  # load in the CLIP model so that we can get the embedding layer
        # self.image_embedder = list(CLIP_model.visual.children())[-1].cuda().eval().float()  #take the embedding layer
        # del CLIP_model

    def forward(self, x):
        # x = self.image_embedder(x) #embed the ROI in CLIP space
        # x /= x.norm(dim=-1, keepdim=True) #normalize the embedded image
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

        self.image_embedder = list(CLIP_model.visual.children())[-1].cuda().eval().float()  # take the embedding layer
        self.text_features = CLIP_model.encode_text(text).detach().to(config.DEVICE).float()
        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

        self.bbox_pred = self._create_regressor(in_channels, 1024, 4).to(config.DEVICE).float()
        del CLIP_model

    def forward(self, x):
        image_features = self.image_embedder(x) #embed the ROI in CLIP space
        image_features /= image_features.norm(dim=-1, keepdim=True) #normalize the embedded image

        # NOTE: scores should be the logits, roi_heads.py takes the logits and performs the softmax.  It also drops all of the first column (assumed to be background)
        scores = (100.0 * image_features @ self.text_features.T)
        bbox_deltas = self.bbox_pred(x)

        expanded_deltas = bbox_deltas # this section will copy the bbox deltas num_classes times
        for _ in range(len(self.text_features)-1):
            expanded_deltas = torch.cat((expanded_deltas, bbox_deltas), 1) # this helps us to perform evaluation without serious surgery to the model

        return scores, expanded_deltas

    def _create_regressor(self, in_channels, projection_dim, out_channels):

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, projection_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(projection_dim, projection_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(projection_dim, out_channels),
        )

class CLIPRPNPredictor(nn.Module):
    """
    CLIP classification + bounding box regression layers
    for Fast R-CNN.
    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, text):
        super(CLIPRPNPredictor, self).__init__()
        CLIP_model, _ = clip.load("RN50", device=config.DEVICE)
        CLIP_model.eval()
        CLIP_model.float()

        self.image_embedder = list(CLIP_model.visual.children())[-1].cuda().eval().float()  # take the embedding layer
        self.text_features = CLIP_model.encode_text(text).detach().to(config.DEVICE).float()
        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

        del CLIP_model

    def forward(self, x):
        image_features = self.image_embedder(x) #embed the ROI in CLIP space
        image_features /= image_features.norm(dim=-1, keepdim=True) #normalize the embedded image

        # NOTE: scores should be the logits, roi_heads.py takes the logits and performs the softmax.  It also drops all of the first column (assumed to be background)
        scores = (100.0 * image_features @ self.text_features.T)

        return scores


import torch.nn.functional as F
def cliprcnn_loss(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for CLIP Faster R-CNN.

    Args:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction='sum',
    )
    box_loss = box_loss / labels.numel()

    return 0, box_loss
