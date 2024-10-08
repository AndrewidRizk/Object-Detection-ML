# VisionMaster: COCO Object Detection with Faster R-CNN

## 📜 Project Overview
**VisionMaster** is an object detection project using **Faster R-CNN** and the **COCO Dataset**. The goal is to train a model that identifies objects in images accurately. This project is built with **PyTorch** and utilizes a subset of the COCO 2017 dataset for efficient training and testing.

## 📂 Dataset Information
The project uses the **COCO 2017** dataset:
- **Train Images**: 118,000 (`train2017`)
- **Validation Images**: 5,000 (`val2017`)
- **Annotations**: Bounding boxes, segmentations, and class labels.

**Download the dataset** from the [COCO Dataset Website](https://cocodataset.org/#download).

## ⚙️ Installation
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. **Set Up a Virtual Environment:**:
```bash
python -m venv coco_env
```
3. **Activate the Environment:**
```bach
.\coco_env\Scripts\activate
```
4.** Install Required Packages:**
 ```bach
pip install -r requirements.txt
```
5. **Download the COCO Dataset and ensure the directory structure:**
 ```bach
D:/ML_Image_Training/coco_dataset/
├── train2017/
├── val2017/
└── annotations_trainval2017/
```

## 🚀 Training the Model
Run the training script:
 ```bach
python train_faster_rcnn.py
```
## ⚙️ Training Options
- Modify the `max_images` parameter in `train_faster_rcnn.py` to limit the number of images used.
- Adjust epochs and batch size as needed.

## 🔄 Resuming Training
Consedering potato computers 🥔. The script automatically saves checkpoints after each epoch in the `checkpoints/` directory. To resume training:
1. Ensure the `checkpoints/` directory has checkpoints.
2. Run:
 ```bach
 python train_faster_rcnn.py
 ```
The script will resume from the latest checkpoint.

## 🙏 Acknowledgements
- COCO Dataset: https://cocodataset.org/#home
- PyTorch: https://pytorch.org/
- Torchvision: https://pytorch.org/vision/stable/index.html
Feel free to contribute, suggest improvements, or report issues!
