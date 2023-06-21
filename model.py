import torch
import torchvision.models as models
import torchvision
from PIL import Image


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        
        # Create a VGG16 network
        self.vgg16 = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=False)
        
        # There are 16 keypoints to detect, each keypoint having 3 atributtes:
        # 1. x coordinate
        # 2. y coordinate
        # 3. a "state" (visible or not) A state of 0 means the joint either does not 
        #   exist or is outside of the image's bounds, 1 denotes a joint that is inside 
        #   of the image but cannot be seen because the part of the object it belongs 
        #   to is not visible in the image, and 2 means the joint was present and visible.
        #   (TODO: this should be one-hot encoded or use embeddings instead of a single number)
        num_out_features = 16 * 3

        # Replace the last layer of the VGG16 network with a linear layer
        self.vgg16.classifier[-1] = torch.nn.Linear(in_features=4096, out_features=num_out_features, bias=True)

    def forward(self, x):

        y_pred = self.vgg16(x)
        return y_pred
    
# Load an image from a file
img = Image.open("path/to/image.jpg")

# Resize image to 224x224
img = torchvision.transforms.functional.resize(img, (224, 224))


# convert it to a tensor
img_tensor = torchvision.transforms.functional.to_tensor(img)

model = Model()
model(img_tensor)