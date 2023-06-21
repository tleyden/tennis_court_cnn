
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
from pytorch_lightning.loggers import TensorBoardLogger

    
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

        # Load the image
        image = Image.open(os.path.join(self.data_path, capture_img_file_path))

        # Convert the image to a tensor
        img_tensor = torchvision.transforms.ToTensor()(image)

        # Remove the alpha channel
        img_tensor = img_tensor[:3]

        # Get a reference to the keypoint annotations
        annotations = capture.annotations
        keypoint_annotations = annotations[0]
        keypoints = keypoint_annotations.values[0].keypoints
        if len(keypoints) != 16:
            raise Exception("Expected 16 keypoints")
        
        # Extract the x,y,state values into a nested list
        flattened_keypoints = [(kp.location[0], kp.location[1], kp.state) for kp in keypoints]

        # Flatten the nested list
        flattened_keypoints = [element for sublist in flattened_keypoints for element in sublist]

        # Convert the list to a tensor
        keypoints_tensor = torch.tensor(flattened_keypoints)
        
        return img_tensor, keypoints_tensor



class LitVGG16(pl.LightningModule):
    
    def __init__(self):
        super().__init__()
        
        # Create a VGG16 network
        self.vgg16 = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=False)
        
        # There are 16 keypoints to detect, each keypoint having 3 atributtes:
        # 1. x coordinate
        # 2. y coordinate
        # 3. a "state" (visible or not) A state of 0 means the joint either does not 
        #   exist or is outside of the image's bounds, 1 denotes a joint that is inside 
        #   of the image but cannot be seen because the part of the object it belongs 
        #   to is not visible in the image, and 2 means the joint was present and visible.
        num_out_features = 16 * 3

        # Replace the last layer of the VGG16 network with a linear layer
        self.vgg16.classifier[-1] = torch.nn.Linear(in_features=4096, out_features=num_out_features, bias=True)

    def forward(self, x):
        y_pred = self.vgg16(x)
        return y_pred
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = torch.nn.functional.mse_loss(y_pred, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    
if __name__ == "__main__":
    dataset = TennisCourtDataset(data_path="/Users/tleyden/Library/Application Support/DefaultCompany/TennisCourt/solo_3")
    train_loader = utils.data.DataLoader(dataset)
    litvgg16 = LitVGG16()
    logger = TensorBoardLogger("logs", name="tennis_court_cnn")
    trainer = pl.Trainer(max_epochs=1, logger=logger)
    trainer.fit(model=litvgg16, train_dataloaders=train_loader)
