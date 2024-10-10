import torch
import torchvision

# Define model architecture similar to the one used during training
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
num_classes = 91  # 80 COCO classes + 1 background class
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Load the saved model weights
# Replace 'faster_rcnn_coco_trained.pth' with the path to your saved file
saved_model_path = "saved_model/faster_rcnn_coco_trained.pth"  # Adjust to your actual file path

try:
    model.load_state_dict(torch.load(saved_model_path))
    model.eval()  # Set the model to evaluation mode for inference
    print("Model loaded successfully. Your training progress is intact!")
except FileNotFoundError:
    print(f"File '{saved_model_path}' not found. Please check if the file exists in the directory.")
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
