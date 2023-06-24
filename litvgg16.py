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
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

        # Normalize the image on the pretrained model's mean and std
        normalized_image = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])(img_tensor)

        return img_tensor, (width, height)
    
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

        # Convert each float -> int
        flattened_keypoints = [int(round(kp)) for kp in flattened_keypoints]
        
        # Convert the flattened keypoints to a list of tuples
        keypoint_pairs = [(flattened_keypoints[i], flattened_keypoints[i + 1]) for i in range(0, len(flattened_keypoints), 2)]

        # Convert the image to opencv format
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        for i, keypoint_pair in enumerate(keypoint_pairs):

            kp_state_pred = kp_states_pred[i]
            if torch.argmax(kp_state_pred).item() != 2:
                # Skip any keypoints that are not visible.  State = 2 means visible
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

    def __init__(self, data_path, transform=None):
        
        self.data_path = data_path

        solo = Solo(data_path=data_path)

        # Preload all frames to allow for random access
        self.solo_frames = [frame for frame in solo.frames()]

    def __len__(self):
        return len(self.solo_frames)

    def __getitem__(self, idx):
        
        solo_frame = self.solo_frames[idx]

        # Each frame has a list of captures, we will just use the first one
        capture = solo_frame.captures[0]

        # Figure out the filepaths for the image
        # TODO: this is a workaround since solo_frame.get_file_path(capture) throws a TypeError: isinstance() arg 2 must be a type, a tuple of types, or a union
        sequence = solo_frame.sequence
        capture_img_file_path = f"sequence.{sequence}/{capture.filename}"

        # Load the image and convert it to the appropriate tensor
        img_tensor, img_size = TennisCourtImageHelper.imagepath2tensor(self.data_path, capture_img_file_path, TennisCourtImageHelper.img_rescale_size)

        # Get a reference to the keypoint annotations
        annotations = capture.annotations
        keypoint_annotations = annotations[0]
        keypoints = keypoint_annotations.values[0].keypoints
        if len(keypoints) != 16:
            raise Exception("Expected 16 keypoints")
        
        # Disabled for now since we are not using the state
        # Extract the x,y,state values into a nested list
        # keypoint_tuples = [(kp.location[0], kp.location[1], kp.state) for kp in keypoints]

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
        
        return img_tensor, keypoints_tensor, keypoint_states_tensor



class LitVGG16(pl.LightningModule):
    
    def __init__(self, num_epochs):

        super().__init__()

        self.num_epochs = num_epochs
        
        # Create a VGG16 network
        self.vgg16 = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True)
        
        # There are 16 keypoints to detect, each keypoint having 3 atributtes:
        # 1. x coordinate
        # 2. y coordinate
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

        # Separate head for categorical output for the 3 different states that each keypoint can have
        # TODO: switch to arrays instead of separate variables
        # self.kp0_state = nn.Linear(25088, 3)
        # self.kp1_state = nn.Linear(25088, 3)
        # self.kp2_state = nn.Linear(25088, 3)
        # self.kp3_state = nn.Linear(25088, 3)
        # self.kp4_state = nn.Linear(25088, 3)
        # self.kp5_state = nn.Linear(25088, 3)
        # self.kp6_state = nn.Linear(25088, 3)
        # self.kp7_state = nn.Linear(25088, 3)
        # self.kp8_state = nn.Linear(25088, 3)
        # self.kp9_state = nn.Linear(25088, 3)
        # self.kp10_state = nn.Linear(25088, 3)
        # self.kp11_state = nn.Linear(25088, 3)
        # self.kp12_state = nn.Linear(25088, 3)
        # self.kp13_state = nn.Linear(25088, 3)
        # self.kp14_state = nn.Linear(25088, 3)
        # self.kp15_state = nn.Linear(25088, 3)
        
        self.kpo_state_layers = [nn.Linear(25088, 3) for i in range(16)]

        print(self.vgg16)




    def forward(self, x):
        vgg_features = self.vgg16(x)
        keypoints_xy = self.continuous_output(vgg_features)

        # TODO: switch to arrays instead of separate variables
        # kp0_state = self.kp0_state(vgg_features)
        # kp1_state = self.kp1_state(vgg_features)
        # kp2_state = self.kp2_state(vgg_features)
        # kp3_state = self.kp3_state(vgg_features)
        # kp4_state = self.kp4_state(vgg_features)
        # kp5_state = self.kp5_state(vgg_features)
        # kp6_state = self.kp6_state(vgg_features)
        # kp7_state = self.kp7_state(vgg_features)
        # kp8_state = self.kp8_state(vgg_features)
        # kp9_state = self.kp9_state(vgg_features)
        # kp10_state = self.kp10_state(vgg_features)
        # kp11_state = self.kp11_state(vgg_features)
        # kp12_state = self.kp12_state(vgg_features)
        # kp13_state = self.kp13_state(vgg_features)
        # kp14_state = self.kp14_state(vgg_features)
        # kp15_state = self.kp15_state(vgg_features)

        kpo_states_pred = [self.kpo_state_layers[i](vgg_features) for i in range(16)]

        #return keypoints_xy, kp0_state, kp1_state, kp2_state, kp3_state, kp4_state, kp5_state, kp6_state, kp7_state, kp8_state, kp9_state, kp10_state, kp11_state, kp12_state, kp13_state, kp14_state, kp15_state

        return keypoints_xy, kpo_states_pred
    
    def training_step(self, batch, batch_idx):
        
        x, keypoints_xy_gt, kp_states = batch

        # keypoints_xy_pred, kp0_state, kp1_state, kp2_state, kp3_state, kp4_state, kp5_state, kp6_state, kp7_state, kp8_state, kp9_state, kp10_state, kp11_state, kp12_state, kp13_state, kp14_state, kp15_state = self(x)
        keypoints_xy_pred, kp_states_pred = self(x)
        
        keypoints_xy_loss = torch.nn.functional.mse_loss(keypoints_xy_pred, keypoints_xy_gt)

        # kp0_loss = torch.nn.functional.cross_entropy(kp0_state, kp_states[:, 0])
        # kp1_loss = torch.nn.functional.cross_entropy(kp1_state, kp_states[:, 1])
        # kp2_loss = torch.nn.functional.cross_entropy(kp2_state, kp_states[:, 2])
        # kp3_loss = torch.nn.functional.cross_entropy(kp3_state, kp_states[:, 3])
        # kp4_loss = torch.nn.functional.cross_entropy(kp4_state, kp_states[:, 4])
        # kp5_loss = torch.nn.functional.cross_entropy(kp5_state, kp_states[:, 5])
        # kp6_loss = torch.nn.functional.cross_entropy(kp6_state, kp_states[:, 6])
        # kp7_loss = torch.nn.functional.cross_entropy(kp7_state, kp_states[:, 7])
        # kp8_loss = torch.nn.functional.cross_entropy(kp8_state, kp_states[:, 8])
        # kp9_loss = torch.nn.functional.cross_entropy(kp9_state, kp_states[:, 9])
        # kp10_loss = torch.nn.functional.cross_entropy(kp10_state, kp_states[:, 10])
        # kp11_loss = torch.nn.functional.cross_entropy(kp11_state, kp_states[:, 11])
        # kp12_loss = torch.nn.functional.cross_entropy(kp12_state, kp_states[:, 12])
        # kp13_loss = torch.nn.functional.cross_entropy(kp13_state, kp_states[:, 13])
        # kp14_loss = torch.nn.functional.cross_entropy(kp14_state, kp_states[:, 14])
        # kp15_loss = torch.nn.functional.cross_entropy(kp15_state, kp_states[:, 15])

        loss = keypoints_xy_loss

        for i, kp_state_pred in enumerate(kp_states_pred):
            kp_loss = torch.nn.functional.cross_entropy(kp_state_pred, kp_states[:, i])
            loss += kp_loss

        # loss = keypoints_xy_loss + kp0_loss + kp1_loss + kp2_loss + kp3_loss + kp4_loss + kp5_loss + kp6_loss + kp7_loss + kp8_loss + kp9_loss + kp10_loss + kp11_loss + kp12_loss + kp13_loss + kp14_loss + kp15_loss

        self.log('train_loss', loss, prog_bar=True)

        # Log the learning rate
        scheduler = self.lr_schedulers()
        current_lr = scheduler.get_last_lr()[0]        
        self.log("learning_rate", current_lr, prog_bar=True)

        # Log the first image of the first batch of each epoch
        if batch_idx == 0:

            first_img_in_batch = x[0]

            # kp0_state_0 = kp0_state[0]
            # kp1_state_0 = kp1_state[0]
            # kp2_state_0 = kp2_state[0]
            # kp3_state_0 = kp3_state[0]
            # kp4_state_0 = kp4_state[0]
            # kp5_state_0 = kp5_state[0]
            # kp6_state_0 = kp6_state[0]
            # kp7_state_0 = kp7_state[0]
            # kp8_state_0 = kp8_state[0]
            # kp9_state_0 = kp9_state[0]
            # kp10_state_0 = kp10_state[0]
            # kp11_state_0 = kp11_state[0]
            # kp12_state_0 = kp12_state[0]
            # kp13_state_0 = kp13_state[0]
            # kp14_state_0 = kp14_state[0]
            # kp15_state_0 = kp15_state[0]

            # kp_states_pred = [kp0_state_0, kp1_state_0, kp2_state_0, kp3_state_0, kp4_state_0, kp5_state_0, kp6_state_0, kp7_state_0, kp8_state_0, kp9_state_0, kp10_state_0, kp11_state_0, kp12_state_0, kp13_state_0, kp14_state_0, kp15_state_0]

            # kp_states_pred_0 = [kp_state_pred[0] for kp_state_pred in kp_states_pred]

            # Get the first batch for all of the kp_state heads.  This will be a list of 16 elements, each element
            # being a tensor of size 3 (for the 3 possible states)
            kp_states_pred_first_batch = []
            for kp_state_pred in kp_states_pred:
                kp_states_pred_first_batch.append(kp_state_pred[0])

            kp_states_gt_first_batch = []
            for kp_state_gt in kp_states[0]:
                predicted_class_index = kp_state_gt
                one_hot_vector = F.one_hot(predicted_class_index, 3, device=self.device)
                kp_states_gt_first_batch.append(one_hot_vector)

            # Convert the tensor to a PIL image
            pil_image = ToPILImage()(first_img_in_batch)

            # Add ground truth and predicted keypoints to the image
            width, height = pil_image.size
            if width != TennisCourtImageHelper.img_rescale_size[0] or height != TennisCourtImageHelper.img_rescale_size[1]:
                raise Exception("Expected image size to be 224x224")
            
            # Show green keypoints for the ground truth and red keypoints for the predicted keypoints
            pil_image_ground_truth = TennisCourtImageHelper.add_keypoints_to_image(pil_image, keypoints_xy_gt[0].tolist(), kp_states_gt_first_batch, color=(0, 255, 0))
            pil_image_predicted = TennisCourtImageHelper.add_keypoints_to_image(pil_image, keypoints_xy_pred[0].tolist(), kp_states_pred_first_batch, color=(0, 0, 255))
            
            wandb.log({f"train_images_epoch_{self.current_epoch}": [wandb.Image(pil_image_ground_truth), wandb.Image(pil_image_predicted)]})

        return loss
    
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
        "data_path", 
        # default="/Users/tleyden/Library/Application Support/DefaultCompany/TennisCourt/solo_19",
        default="/Users/tleyden/Library/Application Support/DefaultCompany/TennisCourt/solo_34",  # missing corners
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
    data_path = args.data_path
    num_epochs = int(args.num_epochs)

    dataset = TennisCourtDataset(data_path=data_path)

    # Create the dataloader and specify the batch size
    train_loader = utils.data.DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True
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

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        max_epochs=num_epochs, 
        logger=wandb_logger, 
        log_every_n_steps=10    # This is only temporarily needed until we train on more data
    )
    trainer.fit(model=litvgg16, train_dataloaders=train_loader)
