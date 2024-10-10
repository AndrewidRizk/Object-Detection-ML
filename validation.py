import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
from coco_dataset_loader import COCODataset
import torchvision.transforms as T
import matplotlib.pyplot as plt
import json

# 1. Define the paths
data_dir = 'D:/ML_Image_Training/coco_dataset'
val_images_dir = os.path.join(data_dir, 'val2017/val2017')
val_annotations_file = os.path.join(data_dir, 'annotations_trainval2017/annotations', 'instances_val2017.json')
model_path = "faster_rcnn_coco_trained.pth"  # Path to the trained model

# 2. Load COCO annotations for validation
coco = COCO(val_annotations_file)

# 3. Define transformations
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. Create the validation dataset
val_dataset = COCODataset(coco, img_dir=val_images_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=lambda x: tuple(zip(*x)))

# 5. Load the trained model
num_classes = 91  # Number of classes in COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)  # No pre-trained weights
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Load the trained weights
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()  # Set the model to evaluation mode

# Convert to Python-native data types for JSON serialization
def convert_to_python_types(coco_predictions):
    # Iterate over the predictions and convert to native types
    for pred in coco_predictions:
        pred["bbox"] = [float(coord) for coord in pred["bbox"]]  # Convert each coordinate to float
        pred["category_id"] = int(pred["category_id"])  # Ensure category_id is an integer
        pred["score"] = float(pred["score"])  # Ensure score is a float
    return coco_predictions

# 6. Define a function to run predictions and print results
def evaluate_on_subset(model, data_loader, device, max_images=100):
    coco_predictions = []
    image_counter = 0  # Keep track of how many images have been processed

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)

            for target, output in zip(targets, outputs):
                image_id = target["image_id"].item()
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()

                # Get the ground truth boxes and labels
                gt_boxes = target["boxes"].cpu().numpy()
                gt_labels = target["labels"].cpu().numpy()

                # Print human-readable results for the current image
                print(f"Image [{image_counter+1}/{max_images}]: ID = {image_id}")
                print(f"  Ground Truth Labels: {gt_labels}")
                print(f"  Ground Truth Boxes: {gt_boxes}")
                print(f"  Predicted Labels: {labels}")
                print(f"  Predicted Boxes: {boxes}")
                print(f"  Predicted Scores: {scores}")
                print("-" * 60)

                # Increment image counter
                image_counter += 1
                if image_counter >= max_images:
                    print(f"Reached limit of {max_images} images. Stopping evaluation.")
                    break

                # Convert predictions to COCO format
                for box, score, label in zip(boxes, scores, labels):
                    x_min, y_min, x_max, y_max = box
                    width = x_max - x_min
                    height = y_max - y_min
                    coco_predictions.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [x_min, y_min, width, height],
                        "score": float(score)
                    })

            # Stop if we have processed the desired number of images
            if image_counter >= max_images:
                break

    return coco_predictions

# 7. Run predictions on a subset of 100 images
print(f"Evaluating the model on a subset of 100 images from the validation set...")
coco_predictions = evaluate_on_subset(model, val_loader, device, max_images=100)

# Convert to native Python types for JSON compatibility
coco_predictions = convert_to_python_types(coco_predictions)

# 8. Convert the predictions to COCO format and save to JSON
predictions_file = "coco_val_subset_predictions.json"
with open(predictions_file, "w") as f:
    json.dump(coco_predictions, f)
print(f"Predictions saved to {predictions_file}")

# 9. Load COCO ground-truth and predicted results for evaluation
coco_gt = COCO(val_annotations_file)  # Ground-truth annotations
coco_dt = coco_gt.loadRes(predictions_file)  # Load predictions

# 10. Perform COCO Evaluation on the subset
from pycocotools.cocoeval import COCOeval
coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
coco_eval.params.imgIds = [pred['image_id'] for pred in coco_predictions]  # Restrict evaluation to the subset images
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
