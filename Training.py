import torch
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.coco as fouc


torch.manual_seed(1)
fo_dataset = foz.load_zoo_dataset("coco-2017", "validation")
fo_dataset.compute_metadata()

session = fo.launch_app(fo_dataset)

# Import functions from the torchvision references we cloned
from engine import train_one_epoch, evaluate
import util




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


