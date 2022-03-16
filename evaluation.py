import torch

from model import create_model
from dataset import get_transforms
import config
import os
import matplotlib.pyplot as plt
from PIL import Image
import skimage
import clip
from utils import evaluate

def eval(item_list, image_dir = None, rpn_score_thresh = 0.05, iou_thresh = .2, conf_thresh = .9, MODEL_TYPE = 'CLIP-FRCNN', MODEL_EPOCH = 311):

    train_transforms, test_transforms = get_transforms()

    rpn_score_thresh = rpn_score_thresh
    iou_thresh = iou_thresh
    conf_thresh = conf_thresh
    item_list = item_list
    MODEL_TYPE = MODEL_TYPE
    MODEL_EPOCH = MODEL_EPOCH


    CHECKPOINT_NAME = f'{MODEL_TYPE}_epoch_{MODEL_EPOCH}.pth'

    # tokenize item list for CLIP
    if item_list[0] != '':
         item_list.insert(0,' ')

    text_tokens = clip.tokenize(["This is " + desc for desc in item_list]).cuda()

    checkpoint = torch.load(CHECKPOINT_NAME)
    clip_frcnn_model = create_model(MODEL_TYPE, classes=text_tokens)

    clip_frcnn_model.load_state_dict(checkpoint)
    clip_frcnn_model.eval()

    clip_frcnn_model.rpn.score_thresh = rpn_score_thresh

    images = []

    if image_dir == None:
        # images in skimage to use and their textual descriptions
        image_dir = skimage.data_dir
        descriptions = {
            # "background": "there is nothing here",
            # "page": "a page of text about segmentation",
            # "chelsea": "a facial photo of a tabby cat",
            "astronaut": "a portrait of an astronaut with the American flag",
            "rocket": "a rocket standing on a launchpad",
            "motorcycle_right": "a red motorcycle standing in a garage",
            "camera": "a person looking at a camera on a tripod",
            "horse": "a black-and-white silhouette of a horse",
            "coffee": "a cup of coffee on a saucer"
        }
    for filename in [filename for filename in os.listdir(image_dir) if filename.endswith(".png") or filename.endswith(".jpg")]:
        name = os.path.splitext(filename)[0]
        if (image_dir == skimage.data_dir):
            if (name not in descriptions):
                continue

        image = Image.open(os.path.join(image_dir, filename)).convert("RGB")

        images.append(test_transforms(image, [0,0,0,0])[0].to(config.DEVICE))

    for number, image in enumerate(images):
        image_out = evaluate(image.unsqueeze(0), item_list, clip_frcnn_model, iou_thresh, conf_thresh)
        plt.imsave(f'./test_images/eval/output_image{number}.jpg',image_out)

if __name__ == "__main__":
    classes = ['flag', 'person', 'helmet', 'basket ball', 'baseball glove', 'astronaut', 'space shuttle', 'camera', 'train', 'Helicopter', 'T-72 tank', 'football helmet', 'soccer ball', 'baseball bat', 'tennis racket', 'cleats shoes', 'boxing gloves', 'man reading a newspaper', 'bike', 'newspaper', 'baby sitting in a swing', 'tree', 'grass', 'sky', 'road']
    image_dir = r'./test_images'
    eval(classes, iou_thresh=.01, conf_thresh=.9, image_dir=image_dir)

