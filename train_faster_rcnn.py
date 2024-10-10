if __name__ == "__main__":
    import os
    import time
    import torch
    import random
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
    num_classes = 91  # Number of classes in COCO
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Device setup
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Optimizer setup
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Define the number of epochs for training
    num_epochs = 10

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

    # ** Define the limit for the number of images to process **
    max_images = 20000  # Manually set the limit to 20,000 images
    images_processed = 0  # Counter to keep track of total images processed

    # ** Calculate the total batches based on the subset size and batch size **
    subset_batches = max_images // data_loader.batch_size  # Calculate batches for 20,000 images
    print(f"Total Batches for this run: {subset_batches}")

    # Training loop with checkpoint saving, progress indicator, and estimated time
    for epoch in range(start_epoch, num_epochs):
        model.train()  # Set model to training mode
        epoch_loss = 0  # Track the cumulative loss over each epoch
        total_batches = subset_batches  # Use calculated subset batches instead of the full dataset

        epoch_start_time = time.time()

        for batch_idx, (images, targets) in enumerate(data_loader):
            try:
                # Check if the limit of 20,000 images has been reached
                images_processed += len(images)
                if images_processed > max_images:
                    print(f"Stopping early: {images_processed} images processed (Limit: {max_images})")
                    break

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

                # ** Update print statement to reflect the correct total_batches value **
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{total_batches}], "
                      f"Progress: {progress:.2f}%, Loss: {losses.item():.4f}, "
                      f"Estimated Time Remaining for Epoch: {estimated_time_remaining:.2f} min")

            except Exception as e:
                print(f"Error during batch processing: {e}")
                continue  # If an error occurs in batch processing, skip to the next batch

        # If the limit has been reached, exit the epoch loop
        if images_processed > max_images:
            print(f"Stopping training after processing {images_processed} images.")
            break

        # Print total loss for the epoch
        epoch_end_time = time.time()
        epoch_duration = (epoch_end_time - epoch_start_time) / 60  # in minutes
        print(f"Epoch [{epoch+1}/{num_epochs}], Total Loss: {epoch_loss:.4f}, "
              f"Epoch Duration: {epoch_duration:.2f} min")

        # Save a checkpoint after every epoch
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
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

    # Save the final trained model
    final_model_path = "faster_rcnn_coco_trained.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved: {final_model_path}")
