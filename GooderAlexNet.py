import pandas as pd, numpy as np
import os
import glob
import wandb
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

base_path = './data'
image_dir = 'jpeg'

train_calc_data = pd.read_csv('{}/csv/calc_case_description_train_set.csv'.format(base_path))
test_calc_data = pd.read_csv('{}/csv/calc_case_description_test_set.csv'.format(base_path))
train_mass_data = pd.read_csv('{}/csv/mass_case_description_train_set.csv'.format(base_path))
test_mass_data = pd.read_csv('{}/csv/mass_case_description_test_set.csv'.format(base_path))

def pre_processing(df):
    df_clean = df.copy()
    df_clean = df.rename(columns = {
        "image file path": "full_mammogram_image_file_path"
    })
    df_clean = df_clean[["full_mammogram_image_file_path", "pathology"]]
    df_clean["pathology"] = df_clean["pathology"].astype("category")
    return df_clean.dropna()

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(5),
    transforms.RandomResizedCrop(size = (224, 224)),
    transforms.ToTensor(),
])

class BreastDataset(Dataset):
    
    def __init__(self, df, transform = train_transforms):
        self.transform = transform
        self.image_pair = []
        for i, row in df.iterrows():
            img_path = []
            
            # Get image path
            full_mammogram_image_path = row["full_mammogram_image_file_path"].rsplit('/')[-2]
            full_mammogram_image_path = "{}/{}/{}".format(base_path, image_dir, full_mammogram_image_path)
            # Glob image
            imgs = glob.glob('{}/*.jpg'.format(full_mammogram_image_path))
            for img in imgs:
                img_path.append(img)
            # Remove duplicate image (if)
            img_path = list(dict.fromkeys(img_path))
            
            # Label image
            pathology = row["pathology"]
            label = 0
            if row["pathology"] == "MALIGNANT":
                label = 1
            elif row["pathology"] == "BENIGN":
                label = 2
            
            for img in img_path:
                self.image_pair.append([img, label])
        
    def __len__(self):
        return len(self.image_pair)

    def __getitem__(self, idx):
        image = Image.open(self.image_pair[idx][0]).convert("RGB")
        image = self.transform(image)
        label = self.image_pair[idx][1]
        return image, label

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        
        # input 3 * 224 * 224
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size = 11, stride = 4, padding = 2), # 64 * 55 * 55
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2, 0), # 64 * 27 * 27
            torch.nn.Conv2d(64, 192, kernel_size = 5, stride = 1, padding = 2), # 192 * 27 * 27
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2, 0), # 192 * 13 * 13
            torch.nn.Conv2d(192, 384, kernel_size = 3, stride = 1, padding = 1), # 384 * 13 * 13
            torch.nn.ReLU(),
            torch.nn.Conv2d(384, 256, kernel_size = 3, stride = 1, padding = 1), # 256 * 13 * 13
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1), # 256 * 13 * 13
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2, 0), # 256 * 6 * 6
        )

        self.dense = torch.nn.Sequential(
            torch.nn.Linear(256 * 6 * 6, 2048),
            torch.nn.ReLU(),
            torch.nn.dropout(p = 0.5),
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(),
            torch.nn.dropout(p = 0.5),
            torch.nn.Linear(512, 3),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.dense(x)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Hyper-parameter
device = "cuda" if torch.cuda.is_available() else "cpu"
epoches = 300
patience = 20
model = Classifier().to(device)
loss_func = nn.CrossEntropyLoss()
batch_size = 64
lr = 5e-5
weight_decay = 0.0005
optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
# momentum = 0.9
# optimizer = torch.optim.SGD(model.parameters(), lr = lr, weight_decay = weight_decay, momentum = momentum)
# Print detail
print(device)
print(f'{model.count_parameters()}')

validation_split = 0.2

# train_data = train_calc_data_clean.copy()
train_calc_data_clean = pre_processing(train_calc_data)
train_mass_data_clean = pre_processing(train_mass_data)
test_calc_data_clean = pre_processing(test_calc_data)
test_mass_data_clean = pre_processing(test_mass_data)

train_data = pd.concat([train_mass_data_clean, train_calc_data_clean], ignore_index = True)
train_set = BreastDataset(train_data, transform = train_transforms)
dataset_size = len(train_set)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
np.random.shuffle(indices)
train_indices, valid_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(valid_indices)

train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, sampler=train_sampler)
valid_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, sampler=valid_sampler)

# test_data = test_calc_data_clean.copy()
test_data = pd.concat([test_calc_data_clean, test_mass_data_clean], ignore_index = True)
test_set = BreastDataset(test_data, transform = test_transforms)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = False)

print(len(train_loader.sampler.indices))
print(len(valid_loader.sampler.indices))
print(len(test_set))

name = "AlexNet-DataAug"

wandb.login(key = os.getenv('WANDB_API_KEY'))
wandb.init(
    project = "Breast-Cancer-Image-Diganosis",
    name = name,
    notes = "Full mammogram image only, with single CNN layer, spliting train data to validate, no data aug",
    tags = ["baseline", "10.5555/2999134.2999257"],
    config = {
        "learning rate": lr,
        "architecture": "AlexNet",
        "epochs": epoches,
        "patience": patience,
        "batch_size": batch_size,
        "image_dim": 224,
        "no_of_params": model.count_parameters(),
        "weighted_decay": weight_decay,
        # "momentum": momentum,
    }
)

best_acc = 0
stale = 0

for epoch in range(epoches):
    
    # Training
    print('Training.....')
    model.train()
    train_loss = 0.0
    train_acc = 0

    for data in tqdm(train_loader):
        
        # Get data from Dataloader
        x_train, y_train = data
        x_train, y_train = x_train.to(device), y_train.to(device)
        
        # Feed the data
        outputs = model(x_train)
        # Clean previous gradient
        optimizer.zero_grad()
        # Calculate loss
        loss = loss_func(outputs, y_train)
        # Calculate grandient
        loss.backward()
        # Update parameter
        optimizer.step()
        
        # Calculate accuracy
        _, pred = torch.max(outputs, 1)
        train_loss += loss.item()
        train_acc += torch.sum(pred == y_train)
        
#         Log training step
        wandb.log({"step_training_loss": loss.item()})
        wandb.log({"step_training_accuracy": torch.sum(pred == y_train)})
    
    # Validation
    print('Validation......')
    model.eval()
    valid_loss = 0.0
    valid_acc = 0
    for data in tqdm(valid_loader):
        # Get the data from Dataloader
        x_valid, y_valid = data
        x_valid, y_valid = x_valid.to(device), y_valid.to(device)
        # Validate the data and no grad needed
        with torch.no_grad():
            outputs = model(x_valid)
        # Calculate loss
        loss = loss_func(outputs, y_valid)
        # Calculate accuracy
        _, pred = torch.max(outputs, 1)
        valid_loss += loss.item()
        valid_acc += torch.sum(pred == y_valid)
        # Log validation step
        wandb.log({"step_training_loss": loss.item()})
        wandb.log({"step_training_accuracy": torch.sum(pred == y_valid)})
    
    # If the model is good, save it.
    if valid_acc > best_acc:
        print(f"Best model found at epoch {epoch}, saving model")
        torch.save(model.state_dict(), f"{name}_best.ckpt")
        best_acc = valid_acc
        stale = 0
    # Else after epoches no improvement, stop.
    else:
        stale += 1
        if stale > patience:
            print(f"No improvement after {patience} epoches, early stopping")
            break
    
    # Test every 5 step
    if epoch % 5 == 0:
        print('Testing....')
        model.eval()
        test_acc = 0

        for data in tqdm(test_loader):
            # Load data
            x_test, y_test = data
            x_test, y_test = x_test.to(device), y_test.to(device)
            # Predict data
            with torch.no_grad():
                outputs = model(x_test)
            _, pred = torch.max(outputs, 1)
            # Calculate accuracy
            test_acc += torch.sum(pred == y_test)
            
    # Print epoch stat
    print("[Epoch: {}/{}] Train Loss is: {:.4f}, Train Acc is: {:.4f}%, Valid Loss is: {:.4f}, Valid Acc is: {:.4f}%".format(
        epoch,
        epoches,
        train_loss / len(train_loader.sampler.indices),
        100 * train_acc / len(train_loader.sampler.indices),
        valid_loss / len(train_loader.sampler.indices),
        100 * valid_acc / len(valid_loader.sampler.indices),
    ))
    if epoch % 5 == 0:
        print("[Epoch: {}/{}] Test Acc is: {:.4f}%".format(
                epoch,
                epoches,
                100 * test_acc / len(test_set),
            ))

    # Log epoch stat
    wandb.log({
        "epoch_train_loss": train_loss / len(train_loader.sampler.indices),
        "epoch_train_acc": 100 * train_acc / len(train_loader.sampler.indices),
        "epoch_valid_loss": valid_loss / len(train_loader.sampler.indices),
        "epoch_valid_acc": 100 * valid_acc / len(valid_loader.sampler.indices),
    })
    if epoch % 5 == 0:
        wandb.log({
            "epoch_test_acc": 100 * test_acc / len(test_set),
        })
    
wandb.finish()

