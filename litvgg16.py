
# Pytorch lightning module for VGG16

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

class TennisCourtImageHelper:

    def __init__(self):
        pass

    @staticmethod
    def imagepath2tensor(data_path_root: str, image_path: str) -> Tensor:

        # Load the image
        image = Image.open(os.path.join(data_path_root, image_path))

        # Convert the image from RGBA (alpha channel) to RGB
        image = image.convert('RGB')

        # Convert the image to a tensor
        img_tensor = torchvision.transforms.ToTensor()(image)

        # Skip this until we figure out how to get the keypoints to match the resized image.
        # According to this post (https://discuss.pytorch.org/t/image-size-for-training-in-vgg/111990/2) it should work with pytorch + vgg
        # Resize image to 224x224
        # img_tensor = torchvision.transforms.functional.resize(img_tensor, (224, 224))

        return img_tensor
    
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
        img_tensor = TennisCourtImageHelper.imagepath2tensor(self.data_path, capture_img_file_path)

        # Get a reference to the keypoint annotations
        annotations = capture.annotations
        keypoint_annotations = annotations[0]
        keypoints = keypoint_annotations.values[0].keypoints
        if len(keypoints) != 16:
            raise Exception("Expected 16 keypoints")
        
        # Extract the x,y,state values into a nested list
        # flattened_keypoints = [(kp.location[0], kp.location[1], kp.state) for kp in keypoints]

        # Extract the x,y values into a nested list
        flattened_keypoints = [(kp.location[0], kp.location[1]) for kp in keypoints]

        # Flatten the nested list
        flattened_keypoints = [element for sublist in flattened_keypoints for element in sublist]

        # Convert the list to a tensor
        keypoints_tensor = torch.tensor(flattened_keypoints)
        
        return img_tensor, keypoints_tensor



class LitVGG16(pl.LightningModule):
    
    def __init__(self, num_epochs):

        super().__init__()

        self.num_epochs = num_epochs
        
        # Create a VGG16 network
        self.vgg16 = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=False)
        
        # There are 16 keypoints to detect, each keypoint having 3 atributtes:
        # 1. x coordinate
        # 2. y coordinate
        # 3. a "state" (visible or not) A state of 0 means the joint either does not 
        #   exist or is outside of the image's bounds, 1 denotes a joint that is inside 
        #   of the image but cannot be seen because the part of the object it belongs 
        #   to is not visible in the image, and 2 means the joint was present and visible.
        # num_out_features = 16 * 3

        # Skip the state for now to make it easier, and only use images where all keypoints are visible
        num_out_features = 16 * 2

        # Replace the last layer of the VGG16 network with a linear layer
        # self.vgg16.classifier[-1] = torch.nn.Linear(in_features=4096, out_features=num_out_features, bias=True)

        # # Freeze the weights of all the CNN layers
        # for param in self.vgg16.features.parameters():
        #     param.requires_grad = False

        # # Verify that the weights are frozen
        # for name, param in self.vgg16.named_parameters():
        #     print(name, param.requires_grad)

        # Redefine the classifier to remove the dropout layers, at least while trying to overfit the network
        self.vgg16.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_out_features)
        )

        print(self.vgg16)

    def forward(self, x):
        y_pred = self.vgg16(x)
        return y_pred
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = torch.nn.functional.mse_loss(y_pred, y)
        self.log('train_loss', loss, prog_bar=True)

        # Log the learning rate
        scheduler = self.lr_schedulers()
        current_lr = scheduler.get_last_lr()[0]        
        self.log("learning_rate", current_lr, prog_bar=True)

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
        default="/Users/tleyden/Library/Application Support/DefaultCompany/TennisCourt/solo_7",
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
