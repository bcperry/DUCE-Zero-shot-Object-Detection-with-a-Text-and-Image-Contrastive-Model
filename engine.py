import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import util
import config
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None, training=True):

    model.train()
    model.backbone.eval() #this is needed to keep the batchnorm layers from the CLIP embedding the same

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    if training:
        header = f"Training Epoch: [{epoch}]"
    else:
        header = f"Testing Epoch: [{epoch}]"
    if training:
        lr_scheduler = None
        if epoch == 0:
            warmup_factor = 1.0 / 1000
            warmup_iters = min(1000, len(data_loader) - 1)

            lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=warmup_factor, total_iters=warmup_iters
            )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)

            #del loss_dict['loss_classifier'] #for the CLIP model, we do not use classification loss

            losses = sum(loss for loss in loss_dict.values())


        loss_value = losses.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict)
            sys.exit(1)
        if training:
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                losses.backward()
                optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

        metric_logger.update(loss=loss_value, **loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

from torch.cuda.amp import autocast

@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        with autocast():
            outputs = model(images)


        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    del outputs
    return coco_evaluator

from torch.utils.tensorboard import SummaryWriter
from model import test_backbone

def train_model(model, train_dataset, validation_dataset, num_epochs=4, MODEL_TYPE='Custom-Vanilla', batch_size = config.BATCH_SIZE, WEIGHTS_NAME = "weights", CONTINUE_TRAINING = False): #'CLIP-FRCNN'  #  Vanilla, Custom-Vanilla, or CLIP-FRCNN):
    # define training and validation data loaders
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=utils.collate_fn,
        drop_last=True
    )

    valid_data_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=utils.collate_fn,
        drop_last=True
    )
    # if MODEL_TYPE=='CLIP-FRCNN':
    #     weight_tester = test_backbone()

    print("Using device %s" % config.DEVICE)

    # move model to the right device
    model.to(config.DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    # check if vanilla has trainable params
    optimizer = torch.optim.Adam(params, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    # create the learning rate schedule
    #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()

    scaler = torch.cuda.amp.GradScaler()

    epoch=0

    if CONTINUE_TRAINING:
        CHECKPOINT_NAME = f'{MODEL_TYPE}_{WEIGHTS_NAME}.pth'
        checkpoint = torch.load(CHECKPOINT_NAME)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch'] + 1
        print(f'Loaded model epoch {epoch}')
    best_loss = 9999
    while epoch < num_epochs:

        # train for one epoch, printing every 10 iterations
        training_metrics = train_one_epoch(model, optimizer, train_data_loader, config.DEVICE, epoch, print_freq=10, training=True, scaler=scaler)

        # if MODEL_TYPE == 'CLIP-FRCNN':  # check that we dont change the weights from the backbone
        #     weight_tester.test(model)

        if epoch % 5 == 0:  #evaluate the mAP of the model every 3 epochs
            # evaluate on the test dataset
            evaluate(model, valid_data_loader, device=config.DEVICE)

        # train for one epoch, printing every 10 iterations
        eval_metrics = train_one_epoch(model, optimizer, valid_data_loader, config.DEVICE, epoch, print_freq=100,
                                  training=False, scaler=scaler)


        #training metrics
        # write to tensorboard
        writer.add_scalar('Learning Rate', training_metrics.meters['lr'].avg, global_step=(epoch))
        writer.add_scalar('Loss/Total Loss', training_metrics.meters['loss'].avg, global_step=(epoch))
        # writer.add_scalar('Loss/Classifier Loss', training_metrics.meters['loss_classifier'].avg, global_step=(epoch))
        writer.add_scalar('Loss/Box Regressor Loss', training_metrics.meters['loss_box_reg'].avg, global_step=(epoch))
        writer.add_scalar('Loss/Objectness Loss', training_metrics.meters['loss_objectness'].avg, global_step=(epoch))
        writer.add_scalar('Loss/RPN Box Regressor Loss', training_metrics.meters['loss_rpn_box_reg'].avg,
                          global_step=(epoch))

        #evaluation metrics
        # write to tensorboard
        writer.add_scalar('Loss/Total Evaluation Loss', eval_metrics.meters['loss'].avg, global_step=(epoch))
        # writer.add_scalar('Loss/Classifier Evaluation Loss', eval_metrics.meters['loss_classifier'].avg, global_step=(epoch))
        writer.add_scalar('Loss/Box Regressor Evaluation Loss', eval_metrics.meters['loss_box_reg'].avg, global_step=(epoch))
        writer.add_scalar('Loss/Objectness Evaluation Loss', eval_metrics.meters['loss_objectness'].avg, global_step=(epoch))
        writer.add_scalar('Loss/RPN Box Regressor Evaluation Loss', eval_metrics.meters['loss_rpn_box_reg'].avg,
                          global_step=(epoch))
        if eval_metrics.meters['loss'].avg <= best_loss:
            total_evaluation_loss = eval_metrics.meters['loss'].avg
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       f'{MODEL_TYPE}_{WEIGHTS_NAME}.pth')
            print(f'Saving epoch {epoch} - Evaluation Loss {total_evaluation_loss}')
            best_loss = eval_metrics.meters['loss'].avg

        epoch += 1

