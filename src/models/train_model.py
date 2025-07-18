import touch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from torchvision import dataset, transforms, models
from torch.utils.data import DataLoader

if torch.backend.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

train_model = datasets.ImageFolder('./data', transform = transform)
load_train = DataLoader(train_model, batch_size = 50, shuffle = True)

def restnet18_train():
    num_classes = len(train_model.classes)
    model = models.resnet18(weights = models.ResNet18_Weights_IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    return model

def balance_class():
    class_weights = compute_class_weight(
        class_weight = 'balanced',
        classes = np.unique(train_model.targets),
        y = np.array(train_model.targets)
    )
    class_weights = torch.tensor(class_weights, dtype = torch.float).to(device)
    return class_weights

