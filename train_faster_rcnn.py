if __name__ == "__main__":
    import os
    import torch
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torch.utils.data import DataLoader
    from pycocotools.coco import COCO
    from coco_dataset_loader import COCODataset
    import torchvision.transforms as T

    # Set up paths
    data_dir = 'D:/ML_Image_Training/coco_dataset'
    train_images_dir = os.path.join(data_dir, 'train2017/train2017')
    annotations_file = os.path.join(data_dir, 'annotations_trainval2017/annotations', 'instances_train2017.json')

    # Load COCO dataset annotations
    coco = COCO(annotations_file)

    # Initialize dataset with transformations (normalize images and convert to tensor)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    coco_dataset = COCODataset(coco, img_dir=train_images_dir, transform=transform)

    # DataLoader to load data in batches during training (using num_workers=0 to avoid multiprocessing errors)
    data_loader = DataLoader(coco_dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=lambda x: tuple(zip(*x)))

    # Load a pre-trained Faster R-CNN model from torchvision
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Modify the model to match the number of classes in COCO (80 classes + 1 for background)
    num_classes = 91  # 80 COCO classes + background class
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Move model to GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Set up optimizer for model training
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Define the number of epochs for training
    num_epochs = 10

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        epoch_loss = 0  # Track the cumulative loss over each epoch

        for images, targets in data_loader:
            # Move images and targets to the same device as the model
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass through the model to compute losses
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass and optimization
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            # Accumulate loss for tracking
            epoch_loss += losses.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
