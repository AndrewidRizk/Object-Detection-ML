if __name__ == "__main__":
    import os
    import time
    import torch
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torch.utils.data import DataLoader, Subset, random_split
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
        T.Resize((300, 300)),  # Resize to 300x300 pixels to reduce computation
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    coco_dataset = COCODataset(coco, img_dir=train_images_dir, transform=transform)

    # Use a subset of the dataset for training (20,000 images)
    subset_indices = list(range(0, 20000))  # Use first 20,000 images for training
    coco_subset = Subset(coco_dataset, subset_indices)

    # Split data into training and validation sets (90% train, 10% validation)
    train_size = int(0.9 * len(coco_subset))
    val_size = len(coco_subset) - train_size
    train_dataset, val_dataset = random_split(coco_subset, [train_size, val_size])

    # DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=lambda x: tuple(zip(*x)))

    # Load and modify the pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    num_classes = 91
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Freeze backbone layers to speed up training
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Device setup
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Optimizer setup
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Define the number of epochs for training
    num_epochs = 5

    # Define checkpoint directory and create it if it doesn't exist
    checkpoint_dir = './checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)  # Create the directory if it doesn't exist
    else:
        print(f"Checkpoint directory '{checkpoint_dir}' already exists.")

    # Automatically find the latest checkpoint if it exists
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]

    if checkpoint_files:
        # Sort checkpoints to get the latest one
        checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        latest_checkpoint = checkpoint_files[-1]
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)

        print(f"Found checkpoints: {checkpoint_files}")
        print(f"Loading latest checkpoint: '{checkpoint_path}'...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from epoch {start_epoch}...")
    else:
        start_epoch = 0
        print("No checkpoint found, starting training from scratch...")

    # Start tracking the time for training
    start_time = time.time()

    # Early stopping setup
    patience = 2  # Stop if the loss doesn't improve for 2 consecutive epochs
    min_loss = float('inf')
    epochs_no_improve = 0

    # Training loop with checkpoint saving, progress indicator, and estimated time
    for epoch in range(start_epoch, num_epochs):
        model.train()  # Set model to training mode
        epoch_loss = 0  # Track the cumulative loss over each epoch
        total_batches = len(train_loader)

        epoch_start_time = time.time()

        for batch_idx, (images, targets) in enumerate(train_loader):
            try:
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

            except Exception as e:
                print(f"Error during batch processing: {e}")
                continue  # If an error occurs in batch processing, skip to the next batch

        # Print total loss for the epoch
        epoch_end_time = time.time()
        epoch_duration = (epoch_end_time - epoch_start_time) / 60  # in minutes
        print(f"Epoch [{epoch+1}/{num_epochs}], Total Loss: {epoch_loss:.4f}, "
              f"Epoch Duration: {epoch_duration:.2f} min")

        # Save a checkpoint after every epoch
        try:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")

        # Early stopping check
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print("Early stopping triggered.")
                break

    # Print total training time
    end_time = time.time()
    total_training_time = (end_time - start_time) / 60  # in minutes
    print(f"Training completed in {total_training_time:.2f} minutes.")

    # Save the final trained model
    final_model_path = "faster_rcnn_coco_trained.pth"
    try:
        torch.save(model.state_dict(), final_model_path)
        print(f"Final model saved: {final_model_path}")
    except Exception as e:
        print(f"Failed to save the final model: {e}")