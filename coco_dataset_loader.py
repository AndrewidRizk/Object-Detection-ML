import torch
from torch.utils.data import Dataset
import cv2
import os

class COCODataset(Dataset):
    def __init__(self, coco, img_dir, transform=None):
        self.coco = coco
        self.img_dir = img_dir
        self.transform = transform
        self.img_ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        # Load the image information
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])

        # Load the image using OpenCV
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load annotations (bounding boxes and labels)
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # Extract bounding boxes and labels
        if len(anns) > 0:
            bboxes = [ann['bbox'] for ann in anns]
            bboxes = torch.as_tensor(bboxes, dtype=torch.float32)

            # Convert bounding box format from [x, y, width, height] to [x_min, y_min, x_max, y_max]
            bboxes[:, 2:] += bboxes[:, :2]

            labels = [ann['category_id'] for ann in anns]
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            # If no annotations are found, return empty tensors
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        # Create target dictionary
        target = {
            "boxes": bboxes,
            "labels": labels,
            "image_id": torch.tensor([img_id])
        }

        # Apply any transformations if provided
        if self.transform:
            image = self.transform(image)

        # At this point, the image is already a tensor if transform is applied
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1)  # Convert from HxWxC to CxHxW

        return image, target
