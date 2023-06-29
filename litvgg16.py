from typing import Any
import torch
from torch import optim, nn, utils, Tensor
import torchvision.models as models
import torchvision
from PIL import Image
import pytorch_lightning as pl
from pysolotools.consumers import Solo
import os
import wandb
from pytorch_lightning.loggers import WandbLogger
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint
from typing import List
from torchvision.transforms import ToPILImage
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import random

class TennisCourtImageHelper:

    # Rescale the image and its associated keypoints to this size
    img_rescale_size = (224, 224)

    def __init__(self):
        pass

    @staticmethod
    def imagepath2tensor(data_path_root: str, image_path: str, rescale_to: tuple[int, int]) -> tuple[Tensor, tuple[int, int]]:

        # Load the image
        image = Image.open(os.path.join(data_path_root, image_path))
        width, height = image.size

        # Convert the image from RGBA (alpha channel) to RGB
        image = image.convert('RGB')

        # Resize image to 224x224
        resized_image = image.resize(rescale_to, Image.LANCZOS)

        # Convert the image to a tensor
        img_tensor = torchvision.transforms.ToTensor()(resized_image)

        # Normalize the image on the pretrained model;s mean and std
        img_tensor_normalized = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])(img_tensor)

        return img_tensor_normalized, img_tensor, (width, height)
    
    @staticmethod
    def rescale_keypoint_coordinates(keypoints: List[tuple[float, float]], orig_size: tuple[int, int], rescaled_size: tuple[int, int]) -> List:
        
        def rescale_keypoint(keypoint: tuple[float, float]) -> tuple[float, float]:
            x, y = keypoint

            # Calculate the scaling factors
            width_scale = rescaled_size[0] / orig_size[0]
            height_scale = rescaled_size[1] / orig_size[1]

            # Transform the keypoint coordinates to the new coordinate system
            resized_x = x * width_scale
            resized_y = y * height_scale

            return (resized_x, resized_y)

        return [rescale_keypoint(keypoint) for keypoint in keypoints]
    
    @staticmethod
    def add_keypoints_to_image(pil_image: Image, flattened_keypoints: List[float], kp_states_pred: List[Tensor], color: tuple[int]) -> Image:

        # The kp states should be a tensor of shape (16, 3).  Each keypoint has a visibility state of 0, 1, or 2, which is one-hot encoded 
        # into a tensor of shape (3). 
        assert kp_states_pred.shape == (16, 3)

        # Convert each float -> int
        flattened_keypoints = [int(round(kp)) for kp in flattened_keypoints]
        
        # Convert the flattened keypoints to a list of tuples
        keypoint_pairs = [(flattened_keypoints[i], flattened_keypoints[i + 1]) for i in range(0, len(flattened_keypoints), 2)]

        # Convert the image to opencv format
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        for i, keypoint_pair in enumerate(keypoint_pairs):

            kp_state_pred = kp_states_pred[i]
            if torch.argmax(kp_state_pred).item() == 0:
                # Skip any keypoints that are not visible.  
                # State = 2 means fully visible, state = 1 means that it's visible but something might be occluding it, but it's still in the image
                continue

            center_coordinates = keypoint_pair  # (x, y) coordinates of the center
            thickness = 2  # Thickness of the circle's outline
            radius = 5  # Radius of the circle
        

            # Draw the circle on the image
            cv2.circle(opencv_image, center_coordinates, radius, color, thickness)

        # Convert the image back to PIL format
        pil_image = Image.fromarray(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))

        return pil_image
         
    
class TennisCourtDataset(torch.utils.data.Dataset):

    def __init__(self, data_paths: List[str], transform=None):
        
        self.solo_frames = []

        # Preload all frames to allow for random access
        for data_path in data_paths:
            solo = Solo(data_path=data_path)
            for frame in solo.frames():
                self.solo_frames.append((frame, data_path))

    def __len__(self):
        return len(self.solo_frames)

    def __getitem__(self, idx):
        
        solo_frame, data_path = self.solo_frames[idx]

        # Each frame has a list of captures, we will just use the first one
        capture = solo_frame.captures[0]

        # Figure out the filepaths for the image
        # TODO: this is a workaround since solo_frame.get_file_path(capture) throws a TypeError: isinstance() arg 2 must be a type, a tuple of types, or a union
        sequence = solo_frame.sequence
        capture_img_file_path = f"sequence.{sequence}/{capture.filename}"

        # Load the image and convert it to the appropriate tensor
        img_tensor, img_tensor_non_normalized, img_size = TennisCourtImageHelper.imagepath2tensor(data_path, capture_img_file_path, TennisCourtImageHelper.img_rescale_size)

        # Get a reference to the keypoint annotations
        annotations = capture.annotations
        keypoint_annotations = annotations[0]
        keypoints = keypoint_annotations.values[0].keypoints
        if len(keypoints) != 16:
            raise Exception("Expected 16 keypoints")
        
        # Extract the x,y values into a nested list
        keypoint_xy_tuples = [(kp.location[0], kp.location[1]) for kp in keypoints]

        # Rescale the keypoints to match the rescaled image
        rescaled_xy_keypoints = TennisCourtImageHelper.rescale_keypoint_coordinates(keypoint_xy_tuples, img_size, TennisCourtImageHelper.img_rescale_size)

        # Flatten the nested list
        flattened_rescaled_keypoints = [element for sublist in rescaled_xy_keypoints for element in sublist]

        # Convert the list to a tensor
        keypoints_tensor = torch.tensor(flattened_rescaled_keypoints)

        # Extract the keypoint states into a tensor
        # A state of 0 means the joint either does not exist or is outside of the image's bounds
        # 1 denotes a joint that is inside of the image but cannot be seen because the part of the object it belongs to is not visible in the image
        # 2 means the joint was present and visible.
        # See https://github.com/Unity-Technologies/com.unity.perception/blob/main/com.unity.perception/Documentation~/HumanPose/TUTORIAL.md
        keypoint_states = [kp.state for kp in keypoints]
        keypoint_states_tensor = torch.tensor(keypoint_states, dtype=torch.long)
        
        return img_tensor, img_tensor_non_normalized, keypoints_tensor, keypoint_states_tensor



class LitVGG16(pl.LightningModule):
    
    def __init__(self, num_epochs):

        super().__init__()

        self.num_epochs = num_epochs
        
        # Create a VGG16 network
        self.vgg16 = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True)
        
        # There are 16 keypoints to detect, each keypoint having 3 atributtes:
        # 1. x coordinate
        # 2. y coordinate
        # TODO: instead of unconstrained x,y coordinates, we should constrain them to the image size. 
        # TODO: maybe a better approach would be to use the normalized coordinates (0-1) and then multiply by the image size (with sigmoid activation)
        num_out_features = 16 * 2

        # Replace the last layer of the VGG16 network with a linear layer
        # self.vgg16.classifier[-1] = torch.nn.Linear(in_features=4096, out_features=num_out_features, bias=True)

        # Freeze the weights of all the CNN layers
        for param in self.vgg16.features.parameters():
            param.requires_grad = False

        # # Verify that the weights are frozen
        # for name, param in self.vgg16.named_parameters():
        #     print(name, param.requires_grad)

        # Redefine the classifier to remove the dropout layers, at least while trying to overfit the network
        self.vgg16.classifier = nn.Identity()

        self.continuous_output = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_out_features)
        )

        # There are 16 keypoints to detect, each keypoint having 3 possible states (see definition above)
        self.kp_states = nn.Linear(25088, 16 * 3)

        print(self.vgg16)

    def forward(self, x):
        vgg_features = self.vgg16(x)
        keypoints_xy = self.continuous_output(vgg_features)

        kp_state_preds = self.kp_states(vgg_features)

        return keypoints_xy, kp_state_preds
    
    def calculate_loss(self, x, keypoints_xy_gt, kp_states_gt):

        x, keypoints_xy_gt, kp_states_gt
        keypoints_xy_pred, kp_states_pred = self(x)
        keypoints_xy_loss = torch.nn.functional.mse_loss(keypoints_xy_pred, keypoints_xy_gt)

        kp_states_pred_reshaped = kp_states_pred.view(-1, 3)
        kp_states_gt_reshaped = kp_states_gt.view(-1)

        kp_loss = torch.nn.functional.cross_entropy(kp_states_pred_reshaped, kp_states_gt_reshaped)

        loss = keypoints_xy_loss + kp_loss
        return loss, (keypoints_xy_pred, kp_states_pred)

    def validation_step(self, batch, batch_idx):

        print("validation_step batch index: ", batch_idx)

        x, img_non_normalized, keypoints_xy_gt, kp_states_gt = batch

        loss, (keypoints_xy_pred, kp_states_pred) = self.calculate_loss(x, keypoints_xy_gt, kp_states_gt)

        self.log('val_loss', loss, prog_bar=True)

        # Log a random image of the first batch of each epoch
        if batch_idx == 0:

            self.superimpose_keypoints(
                img_non_normalized, 
                kp_states_pred, 
                kp_states_gt, 
                keypoints_xy_gt, 
                keypoints_xy_pred, 
                log_prefix="val_images_epoch"
            )

    def training_step(self, batch, batch_idx):
        x, img_non_normalized, keypoints_xy_gt, kp_states_gt = batch
        loss, (keypoints_xy_pred, kp_states_pred) = self.calculate_loss(x, keypoints_xy_gt, kp_states_gt)

        self.log('train_loss', loss, prog_bar=True)

        # Log the learning rate
        scheduler = self.lr_schedulers()
        current_lr = scheduler.get_last_lr()[0]        
        self.log("learning_rate", current_lr, prog_bar=True)

        # Log a random image of the first batch of each epoch
        if batch_idx == 0:

            self.superimpose_keypoints(
                img_non_normalized, 
                kp_states_pred, 
                kp_states_gt, 
                keypoints_xy_gt, 
                keypoints_xy_pred, 
                log_prefix="train_images_epoch"
            )

        return loss
    
    def superimpose_keypoints(self, img_non_normalized, kp_states_pred, kp_states_gt, keypoints_xy_gt, keypoints_xy_pred, log_prefix="train_images_epoch"):

        # Generate a random index within the batch size
        batch_size = img_non_normalized.shape[0]
        random_index = random.randint(0, batch_size - 1)

        img = img_non_normalized[random_index]
        kp_states_pred_random = kp_states_pred[random_index]
        kp_states_gt_random = kp_states_gt[random_index]

        # Convert the tensor to a PIL image
        pil_image = ToPILImage()(img)

        # Add ground truth and predicted keypoints to the image
        width, height = pil_image.size
        if width != TennisCourtImageHelper.img_rescale_size[0] or height != TennisCourtImageHelper.img_rescale_size[1]:
            raise Exception("Expected image size to be 224x224")
        
        # Convert the ground truth keypoint states to one-hot encoding for the first batch
        kp_states_gt_first_batch_one_hot = torch.nn.functional.one_hot(kp_states_gt_random, num_classes=3)

        # Reshape the predicted keypoint states to be 16x3
        kp_states_pred_random = kp_states_pred_random.view(16, 3)
        
        # Show green keypoints for the ground truth and red keypoints for the predicted keypoints
        # TODO: fix bug, it should be passing the ground truth in the first call to add_keypoints_to_image
        pil_image_ground_truth = TennisCourtImageHelper.add_keypoints_to_image(
            pil_image, 
            keypoints_xy_gt[random_index].tolist(), 
            kp_states_gt_first_batch_one_hot, 
            color=(0, 255, 0)
        )
        pil_image_predicted = TennisCourtImageHelper.add_keypoints_to_image(
            pil_image, 
            keypoints_xy_pred[random_index].tolist(), 
            kp_states_pred_random, 
            color=(0, 0, 255)
        )
        
        wandb.log({f"{log_prefix}_{self.current_epoch}": [wandb.Image(pil_image_ground_truth), wandb.Image(pil_image_predicted)]})


    def configure_optimizers(self):
        
        optimizer = optim.Adam(self.parameters(), lr=1e-3)

        # Define a learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.num_epochs, # The maximum number of iterations or epochs before the learning rate is reset. It determines the period of the cosine annealing schedule.
            eta_min=1e-5 # The minimum learning rate. After reaching eta_min, the learning rate will no longer decrease.
        )

        return [optimizer], [scheduler]
    
    
if __name__ == "__main__":

    # Define cli args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_val_data_path", 
        default="/Users/tleyden/Projects/SwingvisionClone/TennisCourtSyntheticDatasets/train_val",  # missing corners
        nargs='?',
        help="Path to the data directory"
    )

    parser.add_argument(
        "num_epochs", 
        default=50,
        nargs='?',
        help="Number of epochs to train for"
    )
    
    # Parse cli args
    args = parser.parse_args()
    train_val_data_path = args.train_val_data_path
    num_epochs = int(args.num_epochs)

    # The training data path should contain one or more solo_ subdirectories
    train_solo_dirs = [os.path.join(train_val_data_path, d) for d in os.listdir(train_val_data_path) if d.startswith("solo_")]
    if len(train_solo_dirs) == 0:
        raise Exception(f"Expected to find one or more solo_ subdirectories in {train_val_data_path}")

    dataset = TennisCourtDataset(data_paths=train_solo_dirs)

    train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

    # Create the train and validation dataloaders
    train_loader = utils.data.DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True
    )
    val_loader = utils.data.DataLoader(
        val_dataset, 
        batch_size=32,
        shuffle=True  # Not strictly needed, but it helps when logging random visualation images to wandb
    )

    # Create the lightning module
    litvgg16 = LitVGG16(num_epochs=num_epochs)

    # Initialize wandb logger
    wandb_logger = WandbLogger(project="tennis_court_cnn")

    # Define a checkpoint callback for saving the model
    checkpoint_callback = ModelCheckpoint(
        dirpath='saved_models',
        filename='model-{epoch:02d}-{train_loss:.4f}',  # Customize the filename as desired
        save_top_k=1,  # Save the best model based on a validation metric
        monitor='train_loss',  # Metric to monitor for saving the best model
        mode='min'  # 'min' or 'max' depending on the monitored metric
    )

    # Create the trainer
    trainer = pl.Trainer(
        # callbacks=[checkpoint_callback], # disable saving checkoints, taking up too much space 
        max_epochs=num_epochs, 
        logger=wandb_logger, 
        log_every_n_steps=10    # This is only temporarily needed until we train on more data
    )
    
    # Train the model
    trainer.fit(
        model=litvgg16, 
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader
    )