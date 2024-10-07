if __name__ == "__main__":
    import os
    import time
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

     # Dataset transformation
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    coco_dataset = COCODataset(coco, img_dir=train_images_dir, transform=transform)

    # DataLoader
    data_loader = DataLoader(coco_dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=lambda x: tuple(zip(*x)))

    # Load and modify the pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    num_classes = 91
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Device setup
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Optimizer setup
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Define the number of epochs for training
    num_epochs = 10

    # Start tracking the time for training
    start_time = time.time()

    # Training loop with checkpoint saving, progress indicator, and estimated time
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        epoch_loss = 0  # Track the cumulative loss over each epoch
        total_batches = len(data_loader)

        epoch_start_time = time.time()

        for batch_idx, (images, targets) in enumerate(data_loader):
            # Start timing the batch
            batch_start_time = time.time()

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

            # Calculate progress percentage
            progress = (batch_idx + 1) / total_batches * 100

            # Calculate elapsed and remaining time
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            batches_left = total_batches - (batch_idx + 1)
            estimated_time_remaining = batches_left * batch_time / 60  # in minutes

            # Print progress percentage and estimated time remaining
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{total_batches}], "
                  f"Progress: {progress:.2f}%, Loss: {losses.item():.4f}, "
                  f"Estimated Time Remaining for Epoch: {estimated_time_remaining:.2f} min")

        # Print total loss for the epoch
        epoch_end_time = time.time()
        epoch_duration = (epoch_end_time - epoch_start_time) / 60  # in minutes
        print(f"Epoch [{epoch+1}/{num_epochs}], Total Loss: {epoch_loss:.4f}, "
              f"Epoch Duration: {epoch_duration:.2f} min")

        # Save a checkpoint after every epoch
        checkpoint_path = f"checkpoint_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    # Print total training time
    end_time = time.time()
    total_training_time = (end_time - start_time) / 60  # in minutes
    print(f"Training completed in {total_training_time:.2f} minutes.")