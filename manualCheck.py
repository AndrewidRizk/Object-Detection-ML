import os
import torch
import torchvision
import random
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# COCO category names
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'TV', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# 1. Load the trained model
def load_model(model_path, num_classes=91, device='cpu'):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

# 2. Image transformation (must match the training transformations)
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Function to run detection on a single image
def detect_objects(model, image_path, device, threshold=0.5):
    # Load the image
    print(f"Processing image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Run the image through the model
    with torch.no_grad():
        outputs = model(image_tensor)

    # Extract boxes, labels, and scores
    boxes = outputs[0]['boxes'].cpu().numpy()
    labels = outputs[0]['labels'].cpu().numpy()
    scores = outputs[0]['scores'].cpu().numpy()

    # Filter out low-confidence detections
    keep_indices = scores >= threshold
    boxes = boxes[keep_indices]
    labels = labels[keep_indices]
    scores = scores[keep_indices]

    return image, boxes, labels, scores

# 4. Function to plot image with detected boxes
def plot_detections(image, boxes, labels, scores, category_names, threshold=0.5):
    # Set up the plot
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    ax = plt.gca()

    # Plot each detected box
    for box, label, score in zip(boxes, labels, scores):
        x_min, y_min, x_max, y_max = box
        width, height = x_max - x_min, y_max - y_min
        color = 'r'  # Use red for bounding boxes

        # Create a rectangle patch
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        # Add label and score text
        plt.text(x_min, y_min - 10, f'{category_names[label]}: {score:.2f}', color=color, fontsize=12, weight='bold')
    
    # Show the plot
    plt.axis('off')
    plt.show()

# 5. Main function to load model and run on a new image
def evaluate_random_image(directory_path, model_path, device='cpu', threshold=0.5):
    # Pick a random image from the directory
    image_files = [f for f in os.listdir(directory_path) if f.endswith('.jpg')]
    if not image_files:
        raise FileNotFoundError(f"No image files found in directory: {directory_path}")

    # Select a random image
    random_image = random.choice(image_files)
    image_path = os.path.join(directory_path, random_image)

    # Load the model
    print("Loading model...")
    model = load_model(model_path, num_classes=91, device=device)

    # Detect objects
    print(f"Running inference on {image_path}...")
    image, boxes, labels, scores = detect_objects(model, image_path, device, threshold=threshold)

    # Convert the image back to a PIL format for visualization
    image = Image.open(image_path).convert("RGB")

    # Plot the detections
    plot_detections(image, boxes, labels, scores, COCO_INSTANCE_CATEGORY_NAMES, threshold=threshold)
    print(f"Detected {len(labels)} objects.")

# 6. Set the paths and run the script on a random image from the directory
directory_path = r"D:\ML_Image_Training\coco_dataset\train2017\train2017"  # Your specified directory
model_path = 'saved_model/faster_rcnn_coco_trained.pth'  # Path to your saved model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Run evaluation on a random image
evaluate_random_image(directory_path, model_path, device, threshold=0.5)
