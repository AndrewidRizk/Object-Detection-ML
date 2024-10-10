from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
from flask_cors import CORS  # Import flask-cors for cross-origin support
import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for rendering
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import io
import torchvision.transforms as T
from werkzeug.utils import secure_filename
import glob
import time

# Flask App Setup
app = Flask(__name__)
CORS(app)  # Enable CORS for the Flask app
app.secret_key = 'supersecretkey'  # Required for session management
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static/uploads')  # Folder to save uploaded files
SLIDESHOW_FOLDER = os.path.join(os.getcwd(), 'static/slideShow')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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

# Load the trained model
model_path = 'saved_model/faster_rcnn_coco_trained.pth'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def load_model(model_path, num_classes=91, device='cpu'):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model(model_path, num_classes=91, device=device)

def detect_and_draw_boxes(image, model, device, threshold=0.5):
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)

    if len(outputs) == 0 or len(outputs[0]['boxes']) == 0:
        return convert_image_to_buffer(image), {}, 0

    # Extract boxes, labels, and scores
    boxes = outputs[0]['boxes'].cpu().numpy()
    labels = outputs[0]['labels'].cpu().numpy()
    scores = outputs[0]['scores'].cpu().numpy()

    # Filter out low-confidence detections
    keep_indices = scores >= threshold
    boxes = boxes[keep_indices]
    labels = labels[keep_indices]
    scores = scores[keep_indices]

    # If no boxes are left after filtering, return the original image
    if len(boxes) == 0:
        return convert_image_to_buffer(image), {}, 0

    # Store detected elements in a dictionary
    detected_elements = {}
    instance_count = 0

    for label, score in zip(labels, scores):
        # Ensure that label indices are within bounds
        if label < len(COCO_INSTANCE_CATEGORY_NAMES):
            label_name = COCO_INSTANCE_CATEGORY_NAMES[label]
        else:
            label_name = "Unknown"

        if label_name not in detected_elements:
            detected_elements[label_name] = []
        detected_elements[label_name].append(f"{score * 100:.2f}%")
        instance_count += 1

    # Draw boxes and labels on the image
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    ax = plt.gca()

    for box, label, score in zip(boxes, labels, scores):
        x_min, y_min, x_max, y_max = box
        width, height = x_max - x_min, y_max - y_min
        color = 'r'  # Use red for bounding boxes

        # Create a rectangle patch
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        # Add label and score text, handle unknown labels
        label_name = COCO_INSTANCE_CATEGORY_NAMES[label] if label < len(COCO_INSTANCE_CATEGORY_NAMES) else "Unknown"
        plt.text(x_min, y_min - 10, f'{label_name}: {score:.2f}', color=color, fontsize=12, weight='bold')

    plt.axis('off')

    # Save the result to an in-memory file as a JPG image
    buf = io.BytesIO()
    plt.savefig(buf, format='jpg')
    buf.seek(0)
    plt.close()
    return buf, detected_elements, instance_count  # Return the buffer and detection info




# Helper function to convert PIL image to an in-memory buffer
def convert_image_to_buffer(image):
    """Converts a PIL image to an in-memory buffer."""
    buf = io.BytesIO()
    image.save(buf, format='JPEG')
    buf.seek(0)
    return buf


@app.route('/', methods=['GET', 'POST'])
def index():
    slideshow_folder = os.path.join(app.static_folder, 'slideshow')
    slideshow_images = [img for img in os.listdir(slideshow_folder) if img.lower().endswith(('png', 'jpg', 'jpeg', 'gif'))]
    if request.method == 'POST':
        
        if 'file' not in request.files:
            return render_template('error.html', message="No file part in the request.")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('error.html', message="No file selected for uploading.")
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Open the image and process it
            image = Image.open(filepath).convert("RGB")
            
            # Correctly unpack three values
            processed_image, detected_elements, total_instances = detect_and_draw_boxes(image, model, device)

            # Save the processed image or the original if no detection
            output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"output_{filename}")
            with open(output_image_path, 'wb') as f:
                f.write(processed_image.getbuffer())

            # Store the detection info in session
            session['total_elements'] = total_instances
            session['detected_elements'] = detected_elements

            # Redirect to view image
            return redirect(url_for('view_image', filename=f'output_{filename}'))

    return render_template('index.html', slideshow_images = slideshow_images)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/view_image/<filename>')
def view_image(filename):
    try:
        # Get the processed image path
        processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        if not os.path.isfile(processed_image_path):
            return render_template('error.html', message="Processed image not found.")

        # Get detection info from session
        total_elements = session.get('total_elements', 0)
        detected_elements = session.get('detected_elements', {})

        # Render the template with the filename and detection info
        return render_template('view_image.html', filename=filename, total_elements=total_elements, detected_elements=detected_elements)

    except Exception as e:
        return render_template('error.html', message=f"An error occurred while displaying the image: {e}")

@app.route('/cleanup')
def cleanup_files():
    # Get all files in the uploads directory
    upload_folder = app.config['UPLOAD_FOLDER']
    
    # Get the list of all files in the directory
    uploaded_files = [f for f in os.listdir(upload_folder) if os.path.isfile(os.path.join(upload_folder, f))]

    # Delete each file
    for file in uploaded_files:
        try:
            os.remove(os.path.join(upload_folder, file))
        except FileNotFoundError:
            continue

    # Clear session data
    session.pop('uploaded_files', None)
    session.pop('total_elements', None)
    session.pop('detected_elements', None)

    # Redirect to the main page after cleanup
    return redirect(url_for('index'))

@app.route('/exit_cleanup', methods=['POST'])
def exit_cleanup():
    # Delete all files in the upload folder
    upload_folder = app.config['UPLOAD_FOLDER']
    
    # Get the list of all files in the directory
    uploaded_files = [f for f in os.listdir(upload_folder) if os.path.isfile(os.path.join(upload_folder, f))]

    # Delete each file
    for file in uploaded_files:
        try:
            os.remove(os.path.join(upload_folder, file))
        except FileNotFoundError:
            continue

    # Clear session data
    session.pop('uploaded_files', None)
    session.pop('total_elements', None)
    session.pop('detected_elements', None)

    return '', 200  # Return an empty response with status code 200

@app.route('/upload_image_from_slideshow')
def upload_image_from_slideshow():
    img_filename = request.args.get('img')
    
    # Construct the path using the 'static/slideshow' directory
    img_path = os.path.join(os.getcwd(), 'static', 'slideshow', img_filename)

    # Debugging: Print the path details
    print(f"Image Filename: {img_filename}")
    print(f"Constructed Image Path: {img_path}")
    print(f"Does file exist? {os.path.isfile(img_path)}")

    if not os.path.isfile(img_path):
        return render_template('error.html', message=f"Image not found: {img_filename} at {img_path}")
    
    # Process the image if found
    image = Image.open(img_path).convert("RGB")
    processed_image, detected_elements, total_instances = detect_and_draw_boxes(image, model, device)

    output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"output_{img_filename}")
    with open(output_image_path, 'wb') as f:
        f.write(processed_image.getbuffer())

    # Store results in the session
    session['total_elements'] = total_instances
    session['detected_elements'] = detected_elements

    return redirect(url_for('view_image', filename=f'output_{img_filename}'))



if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=False)
