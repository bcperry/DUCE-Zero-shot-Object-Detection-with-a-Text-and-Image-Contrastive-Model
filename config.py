import albumentations as A
import cv2
import torch

from albumentations.pytorch import ToTensorV2

DATASET = 'COCO'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_IMAGE_ARCHITECTURE = "RN50"
NUM_WORKERS = 3
TRAIN_TEST_SPLIT = .8  # training ratio of total
BATCH_SIZE = 2
INPUT_RESOLUTION = 224
NUM_CLASSES = 91
LEARNING_RATE = 5e-3
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 1000
PIN_MEMORY = True
LOAD_MODEL = True
SAVE_MODEL = True

FILENAME = "Full_Model.pth.tar"
DATA_DRIVE = "C:/Data_drive/Data/VOC"
TRAIN_IMG_DIR = 'C:\\Data_drive\\Data\\COCO_2017_VAL\\train2017\\train2017\\'
TRAIN_LABEL_FILE = "C:\\Data_drive\\Data\\COCO_2017_VAL\\annotations_trainval2017\\annotations\\instances_train2017.json"
TRAIN_CSV_FILE = "/train.csv"
TEST_CSV_FILE = "/test.csv"
WEIGHT_FILE = FILENAME

# CLIP normalization of images
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)

PASCAL_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

COCO_LABELS = ['person',
               'bicycle',
               'car',
               'motorcycle',
               'airplane',
               'bus',
               'train',
               'truck',
               'boat',
               'traffic light',
               'fire hydrant',
               'street sign',
               'stop sign',
               'parking meter',
               'bench',
               'bird',
               'cat',
               'dog',
               'horse',
               'sheep',
               'cow',
               'elephant',
               'bear',
               'zebra',
               'giraffe',
               'hat',
               'backpack',
               'umbrella',
               'shoe',
               'eye glasses',
               'handbag',
               'tie',
               'suitcase',
               'frisbee',
               'skis',
               'snowboard',
               'sports ball',
               'kite',
               'baseball bat',
               'baseball glove',
               'skateboard',
               'surfboard',
               'tennis racket',
               'bottle',
               'plate',
               'wine glass',
               'cup',
               'fork',
               'knife',
               'spoon',
               'bowl',
               'banana',
               'apple',
               'sandwich',
               'orange',
               'broccoli',
               'carrot',
               'hot dog',
               'pizza',
               'donut',
               'cake',
               'chair',
               'couch',
               'potted plant',
               'bed',
               'mirror',
               'dining table',
               'window',
               'desk',
               'toilet',
               'door',
               'tv',
               'laptop',
               'mouse',
               'remote',
               'keyboard',
               'cell phone',
               'microwave',
               'oven',
               'toaster',
               'sink',
               'refrigerator',
               'blender',
               'book',
               'clock',
               'vase',
               'scissors',
               'teddy bear',
               'hair drier',
               'toothbrush',
               'hair brush',
               ]
