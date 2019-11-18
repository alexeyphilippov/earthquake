import os
import torch.nn as nn
import torch
from torchvision import datasets, models, transforms

path_for_saved_model = './models'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == "__name__":
    model_name = 'model.pt'
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features

    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join(path_for_saved_model, model_name), map_location='cpu'))
    model.eval()

