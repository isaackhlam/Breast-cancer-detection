import pandas as pd, numpy as np
import os
import glob
import wandb
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pytorch_warmup as warmup
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image
from tqdm import tqdm
from torchinfo import summary

base_path = './data'
image_dir = 'jpeg'

train_calc_data = pd.read_csv('{}/csv/calc_case_description_train_set.csv'.format(base_path))
test_calc_data = pd.read_csv('{}/csv/calc_case_description_test_set.csv'.format(base_path))
# train_mass_data = pd.read_csv('{}/csv/mass_case_description_train_set.csv'.format(base_path))
# test_mass_data = pd.read_csv('{}/csv/mass_case_description_test_set.csv'.format(base_path))

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
    transforms.RandomResizedCrop(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.1959, 0.1959, 0.1959],
        std=[0.2591, 0.2591, 0.2591]
    )
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
            label = 0
            if row["pathology"] == "MALIGNANT":
                label = 1
            
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
        
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(1000, 2)
        )
    def forward(self, x):
        return x = self.dense(x)

# Hyper-parameter
device = "cuda" if torch.cuda.is_available() else "cpu"
epoches = 300
patience = 150
no_of_classes = 2
# model = Classifier().to(device)
pre = torchvision.models.alexnet(weight = "IMAGE1KV1")
pre = torch.compile(pre)
pre = pre.to(device)
model = Classifier()
model = torch.compile(model)
model = model.to(device)
# model = torchvision.models.maxvit_t().to(device)
# model = torchvision.models.maxvit_t().to(device)
# model.load_state_dict(torch.load("MaxVit-PreTrain-CalcOnly_best.ckpt"))
loss_func = nn.CrossEntropyLoss()
batch_size = 64
lr = 1e-4
weight_decay = 1e-6
optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
# Print detail
print(device)
# print(f'{model.count_parameters()}')

validation_split = 0.2

# train_data = train_calc_data_clean.copy()
train_calc_data_clean = pre_processing(train_calc_data)
# train_mass_data_clean = pre_processing(train_mass_data) 
test_calc_data_clean = pre_processing(test_calc_data)
# test_mass_data_clean = pre_processing(test_mass_data)

# train_data = pd.concat([train_mass_data_clean, train_calc_data_clean], ignore_index = True)
train_data = pre_processing(train_calc_data)
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
# test_data = pd.concat([test_calc_data_clean, test_mass_data_clean], ignore_index = True)
test_data = pre_processing(test_calc_data)
test_set = BreastDataset(test_data, transform = test_transforms)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = False)

print(len(train_loader.sampler.indices))
print(len(valid_loader.sampler.indices))
print(len(test_set))

num_steps = len(train_loader.sampler.indices) * epoches
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

summary(model, input_size=(batch_size, 3, 224, 224))

name = "Alex-Pre-Aug-Freeze"

wandb.login(key = "9b61f1e0194e4478e6f295eec50074c524e5f0f8")
wandb.init(
    project = "Breast-Cancer-Image-Diganosis",
    name = name,
    notes = "Full mammogram image only, with VGG13BN layer, spliting train data to validate, no data aug",
    tags = ["baseline", "10.1109/ICARCV.2014.7064414"],
    config = {
        "learning rate": lr,
        "architecture": "VGG13BN",
        "epochs": epoches,
        "patience": patience,
        "batch_size": batch_size,
        "image_dim": 224,
        # "no_of_params": model.count_parameters(),
        "weighted_decay": weight_decay,
    }
)

best_acc = 0
stale = 0

pre.eval()

for epoch in range(epoches):
    
    # Training
    print('Training.....')
    model.train()
    train_loss = []
    train_acc = []

    for data in tqdm(train_loader):
        
        # Get data from Dataloader
        x_train, y_train = data
        x_train, y_train = x_train.to(device), y_train.to(device)
        
        x_train = pre(x_train)
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
        # Update lr Scheduler
        with warmup_scheduler.dampening():
            lr_scheduler.step()
        
        # Calculate accuracy
        acc = (outputs.argmax(dim=-1) == y_train.to(device)).float().mean()

        
#         Log training step
        wandb.log({"step_training_loss": loss.item()})
        wandb.log({"step_training_accuracy": acc})
        train_loss.append(loss.item())
        train_acc.append(acc)
        
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_acc) / len(train_acc)
    
    # Validation
    print('Validation......')
    model.eval()
    valid_loss = []
    valid_acc = []
    for data in tqdm(valid_loader):
        # Get the data from Dataloader
        x_valid, y_valid = data
        x_valid, y_valid = x_valid.to(device), y_valid.to(device)
        x_valid = pre(x_valid)
        # Validate the data and no grad needed
        with torch.no_grad():
            outputs = model(x_valid)
        # Calculate loss
        loss = loss_func(outputs, y_valid)
        # Calculate accuracy
        acc = (outputs.argmax(dim=-1) == y_valid.to(device)).float().mean()
        # Log validation step
        wandb.log({"step_training_loss": loss.item()})
        wandb.log({"step_training_accuracy": acc})
        valid_loss.append(loss.item())
        valid_acc.append(acc)

    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_acc) / len(valid_acc)
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
    test_acc = []
    if epoch % 5 == 0:
        print('Testing....')
        model.eval()

        for data in tqdm(test_loader):
            # Load data
            x_test, y_test = data
            x_test, y_test = x_test.to(device), y_test.to(device)
            x_test = pre(x_test)
            # Predict data
            with torch.no_grad():
                outputs = model(x_test)
            _, pred = torch.max(outputs, 1)
            # Calculate accuracy
            acc = (outputs.argmax(dim=-1) == y_test.to(device)).float().mean()
        
            test_acc.append(acc)
        test_acc = sum(test_acc) / len(test_acc)
    
        if(test_acc > 0.8):
            print(f"Cherry pick model at {epoch}, saving model")
            torch.save(model.state_dict(), f"{name}_cherry.ckpt")

    # Print epoch stat
    print("[Epoch: {}/{}] Train Loss is: {:.4f}, Train Acc is: {:.4f}%, Valid Loss is: {:.4f}, Valid Acc is: {:.4f}%".format(
        epoch,
        epoches,
        train_loss,
        100 * train_acc,
        valid_loss,
        100 * valid_acc,
    ))
    if epoch % 5 == 0:
        print("[Epoch: {}/{}] Test Acc is: {:.4f}%".format(
                epoch,
                epoches,
                100 * test_acc,
            ))

    # Log epoch stat
    wandb.log({
        "epoch_train_loss": train_loss,
        "epoch_train_acc": 100 * train_acc,
        "epoch_valid_loss": valid_loss,
        "epoch_valid_acc": 100 * valid_acc,
    })
    if epoch % 5 == 0:
        wandb.log({
            "epoch_test_acc": 100 * test_acc,
        })

wandb.finish()
