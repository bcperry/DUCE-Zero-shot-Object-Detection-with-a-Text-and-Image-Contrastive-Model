import datetime
import errno
import os
import random
import time
from collections import defaultdict, deque

import torch
import torch.distributed as dist
from torch import nn

from typing import Iterable, Callable
from torch import Tensor

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import torchvision.transforms as transforms
import config

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes

def get_transforms(Normalize = False):

    if Normalize:
        test_transforms = Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean=config.MEAN, std=config.STD), ])
        train_transforms = Compose(
            [transforms.ToTensor(), transforms.RandomHorizontalFlip(0.5), transforms.Normalize(mean=config.MEAN, std=config.STD), ])
    else:
        train_transforms = Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(0.5)])
        test_transforms = Compose([transforms.ToTensor()])
    return train_transforms, test_transforms

def collate_fn(batch):
    valid = []
    replace = []
    for image in range(len(batch)):
        if len(batch[image][1]['boxes']) != 0: #make sure the image has some bbox
            valid.append(image)
        else:
            replace.append(image)
    if len(valid) != len(batch):
        for bad in replace:
            batch[bad] = batch[random.choice(valid)]
    return tuple(zip(*batch))

class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = torch.empty(0)

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features = output
        return fn

    def forward(self, x: Tensor):
        _ = self.model(x)
        return self._features


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.inference_mode():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)





def convert_torch_predictions(preds, det_id, s_id, w, h, classes, labelmap = None):
    import fiftyone as fo
    import fiftyone.utils.coco as fouc

    # labelmap is a dictionary mapping from label integer to label name

    # Convert the outputs of the torch model into a FiftyOne Detections object
    dets = []
    for bbox, label, score in zip(
            preds["boxes"].cpu().detach().numpy(),
            preds["labels"].cpu().detach().numpy(),
            preds["scores"].cpu().detach().numpy()
    ):
        # Parse prediction into FiftyOne Detection object
        x0, y0, x1, y1 = bbox

        if labelmap is not None:
            label_class = labelmap[int(label)] # get the class name
            if label_class in classes: # check if the label is in the class list
                label = classes.index(label_class) # convert the label to the appropriate class id
            else:
                label = 0 # consider this a background class

        coco_obj = fouc.COCOObject(det_id, s_id, int(label), [x0, y0, x1 - x0, y1 - y0])
        det = coco_obj.to_detection((w, h), classes)
        det["confidence"] = float(score)
        dets.append(det)
        det_id += 1

    detections = fo.Detections(detections=dets)

    return detections, det_id


def add_detections(model, torch_dataset, view, field_name="predictions", labelmap=None, PRED_CLUSTERING=False, eps = 30):
    import fiftyone as fo

    # Run inference on a dataset and add results to FiftyOne
    torch.set_num_threads(1)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using device %s" % device)

    model.eval()
    model.to(device)
    image_paths = torch_dataset.img_paths
    classes = torch_dataset.classes
    det_id = 0

    with fo.ProgressBar() as pb:
        for img, targets in pb(torch_dataset):
            # Get FiftyOne sample indexed by unique image filepath
            img_id = int(targets["image_id"][0])
            img_path = image_paths[img_id]
            sample = view[img_path]
            s_id = sample.id
            w = sample.metadata["width"]
            h = sample.metadata["height"]

            if PRED_CLUSTERING:
                preds = model(img.unsqueeze(0).to(device))
                # Combine bboxes with clustering
                preds = evaluate_custom(image = img.unsqueeze(0),
                                        labels = classes,
                                        preds = preds,
                                        iou_thresh = 1,
                                        conf_thresh = 0,
                                        show = False,
                                        weighted=True,
                                        eps = eps)
                boxes = []
                labels = []
                scores = []

                boxes.append(torch.tensor([box[2:] for box in preds]))
                labels.append(torch.tensor([box[0] for box in preds]))
                scores.append(torch.tensor([box[1] for box in preds]))

                final_preds = {'boxes': boxes[0],
                               'labels': labels[0],
                               'scores': scores[0]}

                detections, det_id = convert_torch_predictions(
                    final_preds,
                    det_id,
                    s_id,
                    w,
                    h,
                    classes,
                )
            else:
                preds = model(img.unsqueeze(0).to(device))[0]

                detections, det_id = convert_torch_predictions(
                    preds,
                    det_id,
                    s_id,
                    w,
                    h,
                    classes,
                    labelmap,
                )



            sample[field_name] = detections
            sample.save()

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Video explanation of this function:
    https://youtu.be/XXYG5ZWtjj0

    This function calculates intersection over union (iou) given pred boxes
    and target boxes.

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Video explanation of this function:
    https://youtu.be/YDkjWEN8jNA

    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list
    if len(bboxes)== 7:
        class_prob_idx = 1
        box_prob_idx = 2
        box_start_idx = 3
    else:
        class_prob_idx = 1
        box_prob_idx = 1
        box_start_idx = 2

    bboxes = [box for box in bboxes if box[box_prob_idx] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[box_prob_idx], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
               or intersection_over_union(
                torch.tensor(chosen_box[box_start_idx:]),
                torch.tensor(box[box_start_idx:]),
                box_format=box_format,
            )
               < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms



def plot_image(image, boxes, class_labels, show = True):
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image

    ax.imshow(im)

    # Create a Rectangle patch
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        class_conf = np.round(box[1],2)
        box = box[2:]
        upper_left_x = box[0]
        upper_left_y = box[1]
        w = box[2] - box[0]
        h = box[3] - box[1]

        rect = Rectangle(
            (upper_left_x, upper_left_y),
            w,
            h,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.text(
            upper_left_x,
            upper_left_y,
            s=class_labels[int(class_pred)] + ": " + str(class_conf),
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))

    if show:
        plt.show()
    else:
        plt.close()
    return im

def evaluate(image, labels, preds, iou_thresh, conf_thresh, show = True):


    #organize the output for NMS and plotting
    bboxes = preds[0]['boxes'].detach().cpu().numpy()
    bboxes = np.insert(bboxes, 0, preds[0]['labels'].detach().cpu().numpy(), axis=1)
    bboxes = np.insert(bboxes, 1, preds[0]['scores'].detach().cpu().numpy(), axis=1)
    bboxes = list(bboxes)

    nms_boxes = non_max_suppression(bboxes, iou_thresh, conf_thresh)

    im = plot_image(image.detach().cpu()[0].permute(1,2,0), nms_boxes, labels, show)
    return im

def evaluate_custom(image = None, labels = None, preds = None, iou_thresh = 0.2, conf_thresh = 0.8, show = True, weighted=True, eps=50):

    import pandas as pd

    #organize the output for NMS and plotting
    bboxes = preds[0]['boxes'].detach().cpu().numpy()
    bboxes = np.insert(bboxes, 0, preds[0]['labels'].detach().cpu().numpy(), axis=1)
    bboxes = np.insert(bboxes, 1, preds[0]['scores'].detach().cpu().numpy(), axis=1)
    bboxes = pd.DataFrame(bboxes, columns=('class_pred', 'prob_score', 'x1', 'y1', 'x2', 'y2'))

    #drop low confidence boxes
    bboxes.drop(bboxes[bboxes.prob_score < conf_thresh].index, inplace=True)
    bboxes['x_mid'] = (bboxes.x2 - bboxes.x1)/2 + bboxes.x1
    bboxes['y_mid'] = (bboxes.y2 - bboxes.y1)/2 + bboxes.y1


    bboxes = average_bboxes(bboxes, weighted=weighted, eps=eps)

    bboxes = non_max_suppression(bboxes, iou_thresh, conf_thresh)

    #remove the box probability and show only class probability
    final_box_list = []
    for box in bboxes:
        box.pop(2)
        final_box_list.append(box)
    if show:
        im = plot_image(image.detach().cpu()[0].permute(1,2,0), final_box_list, labels, show)
        return im
    else:
        return final_box_list

def average_bboxes(bboxes, weighted = True, eps=30):

    from sklearn.cluster import DBSCAN
    unique_preds = bboxes.class_pred.unique()
    combined_preds = []
    for pred in unique_preds:

        class_bboxes = bboxes[bboxes.class_pred == pred].copy()
        db = DBSCAN(eps=eps).fit(class_bboxes[['x_mid', 'y_mid']])
        class_bboxes['cluster'] = db.labels_


        for group in class_bboxes.cluster.unique():
            clustered_bboxes = class_bboxes[class_bboxes.cluster == group]
            label = pred
            class_confidence = clustered_bboxes.prob_score.max()
            box_confidence = clustered_bboxes.prob_score.mean()
            if weighted:

                x1 = sum(clustered_bboxes.x1 * clustered_bboxes.prob_score) / clustered_bboxes.prob_score.sum()
                x2 = sum(clustered_bboxes.x2 * clustered_bboxes.prob_score) / clustered_bboxes.prob_score.sum()
                y1 = sum(clustered_bboxes.y1 * clustered_bboxes.prob_score) / clustered_bboxes.prob_score.sum()
                y2 = sum(clustered_bboxes.y2 * clustered_bboxes.prob_score) / clustered_bboxes.prob_score.sum()

                combined_preds.append([label, class_confidence, box_confidence, x1, y1, x2, y2])

            else:
                x1 = clustered_bboxes.x1.mean()
                x2 = clustered_bboxes.x2.mean()
                y1 = clustered_bboxes.y1.mean()
                y2 = clustered_bboxes.y2.mean()

                combined_preds.append([label, class_confidence, box_confidence, x1, y1, x2, y2])

    return combined_preds

