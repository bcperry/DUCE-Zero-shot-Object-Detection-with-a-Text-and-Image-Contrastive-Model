import torch
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.coco as fouc
from dataset import FiftyOneTorchDataset
from model import create_model

torch.manual_seed(1)
fo_dataset = foz.load_zoo_dataset("coco-2017", "validation")
fo_dataset.compute_metadata()

session = fo.launch_app(fo_dataset)

# Import functions from the torchvision references we cloned
from engine import train_one_epoch, evaluate
import utils


def do_training(model, torch_dataset, torch_dataset_test, num_epochs=4):
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        torch_dataset, batch_size=2, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        torch_dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using device %s" % device)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)

        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)


from fiftyone import ViewField as F

vehicles_list = ["car", "truck", "bus"]
vehicles_view = fo_dataset.filter_labels("ground_truth",
        F("label").is_in(vehicles_list))

print(len(vehicles_view))

session.view = vehicles_view

# From the torchvision references we cloned
import transforms as T

train_transforms = T.Compose([T.ToTensor(), T.RandomHorizontalFlip(0.5)])
test_transforms = T.Compose([T.ToTensor()])

# split the dataset in train and test set
train_view = vehicles_view.take(500, seed=51)
test_view = vehicles_view.exclude([s.id for s in train_view])

# use our dataset and defined transformations
torch_dataset = FiftyOneTorchDataset(train_view, train_transforms,
        classes=vehicles_list)
torch_dataset_test = FiftyOneTorchDataset(test_view, test_transforms,
        classes=vehicles_list)

model = create_model(model_type='Custom-Vanilla',
                     num_classes=(len(vehicles_list)+1))

do_training(model, torch_dataset, torch_dataset_test, num_epochs=4)

import fiftyone as fo


def convert_torch_predictions(preds, det_id, s_id, w, h, classes):
    # Convert the outputs of the torch model into a FiftyOne Detections object
    dets = []
    for bbox, label, score in zip(
            preds["boxes"].cpu().detach().numpy(),
            preds["labels"].cpu().detach().numpy(),
            preds["scores"].cpu().detach().numpy()
    ):
        # Parse prediction into FiftyOne Detection object
        x0, y0, x1, y1 = bbox
        coco_obj = fouc.COCOObject(det_id, s_id, int(label), [x0, y0, x1 - x0, y1 - y0])
        det = coco_obj.to_detection((w, h), classes)
        det["confidence"] = float(score)
        dets.append(det)
        det_id += 1

    detections = fo.Detections(detections=dets)

    return detections, det_id


def add_detections(model, torch_dataset, view, field_name="predictions"):
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

            # Inference
            preds = model(img.unsqueeze(0).to(device))[0]

            detections, det_id = convert_torch_predictions(
                preds,
                det_id,
                s_id,
                w,
                h,
                classes,
            )

            sample[field_name] = detections
            sample.save()

add_detections(model, torch_dataset_test, fo_dataset, field_name="predictions")

results = fo.evaluate_detections(
    test_view,
    "predictions",
    classes=["car", "bus", "truck"],
    eval_key="eval",
    compute_mAP=True
)

results.mAP()
results.print_report()

results_interclass = fo.evaluate_detections(
    test_view,
    "predictions",
    classes=["car", "bus", "truck"],
    compute_mAP=True,
    classwise=False
)

results_interclass.plot_confusion_matrix()

session.view = test_view.sort_by("eval_fp", reverse=True)

session.view = test_view.sort_by("eval_fp", reverse=True)


