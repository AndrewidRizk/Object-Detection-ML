from pycocotools.coco import COCO
from coco_dataset_loader import COCODataset
import os
import matplotlib.pyplot as plt
import cv2
import random



# Define paths
data_dir = 'D:/ML_Image_Training/coco_dataset'
train_images_dir = os.path.join(data_dir, 'train2017/train2017')
annotations_file = os.path.join(data_dir, 'annotations_trainval2017/annotations', 'instances_train2017.json')

# Load COCO dataset annotations
coco = COCO(annotations_file)

# Initialize dataset
coco_dataset = COCODataset(coco, img_dir=train_images_dir)

# Test loading data
for i in range(5):
    idx = random.randint(0, len(coco_dataset) - 1)
    image, bboxes = coco_dataset[idx]
    # Convert from tensor to numpy array for plotting
    image = image.permute(1, 2, 0).numpy()

    # Draw bounding boxes using OpenCV
    for bbox in bboxes:
        x, y, w, h = map(int, bbox)
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the image
    plt.imshow(image)
    plt.show()
