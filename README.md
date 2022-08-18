# No-Shot Learning with CLIP
## Model and Capability Demonstration Repository

---
## Blaine Perry, Blaine Kuehnert, Jarek Ingros, Jonathan Dencker
---

### Directory Structure
```
|--- noshot_model_repo*
  |--- CLIP-RPN_rpn_full_training_epoch_30.pth**
  |--- yolov5s.pt**
  |--- TORCH_CLIP_FRCNN_Trainable***
    |--- dataset_analysis
      |--- dataset_analysis.xlsx
    |--- coco
      |--- coco_eval.py
      |--- coco_utils.py
    |--- fiftyOne
      |--- Imagenet_dataset_fiftyOne.ipynb
      |--- dataset.py
    |--- utils
      |--- config.py
      |--- custom_roi_heads.py
      |--- custom_rpn.py
      |--- engine.py
      |--- evaluation.py
      |--- model.py
      |--- Training.py
      |--- transforms.py
      |--- util.py   
    |--- CLIP-FRCNN-DEMO
    |--- CLIP-FRCNN-pytorch_evaluation
    |--- CLIP-FRCNN-pytorch_evaluation_YOLO
    |--- CLIP-FRCNN-pytorch_training
    |--- environment.yml
    |--- README.md
    |--- .gitignore
    |--- .getattributes
```
* Due to the size of model .pt and .pth files, we opted to create a parent directory in which the model files were stored. This repo sits at the root of that directory as well.
** These model files can be replaced with your own trained model files. They are available for download at ...
*** This is the root of the repository
