import pandas as pd, numpy as np
import os
import glob
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(0.5),
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
            torch.nn.Linear(256 * 6 * 6, 4096),
            torch.nn.ReLU(),
            torch.nn.dropout(p = 0.5),
            torch.nn.Linear(4096, 1024),
            torch.nn.ReLU(),
            torch.nn.dropout(p = 0.5),
            torch.nn.Linear(1024, 3),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.dense(x)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class basic_block(nn.Module):
    # 輸出通道乘的倍數
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, downsample):
        super(basic_block, self).__init__()      
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 在 shortcut 時，若維度不一樣，要更改維度
        self.downsample = downsample 


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
      
class bottleneck_block(nn.Module):
    # 輸出通道乘的倍數
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, downsample):
        super(bottleneck_block, self).__init__()      
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        # 在 shortcut 時，若維度不一樣，要更改維度
        self.downsample = downsample 


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
      
class ResNet(nn.Module):
    def __init__(self, net_block, layers, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.net_block_layer(net_block, 64, layers[0])
        self.layer2 = self.net_block_layer(net_block, 128, layers[1], stride=2)
        self.layer3 = self.net_block_layer(net_block, 256, layers[2], stride=2)
        self.layer4 = self.net_block_layer(net_block, 512, layers[3], stride=2)

        self.avgpooling = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * net_block.expansion, num_classes)

        # 參數初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)        

    def net_block_layer(self, net_block, out_channels, num_blocks, stride=1):
        downsample = None

      # 在 shortcut 時，若維度不一樣，要更改維度
        if stride != 1 or self.in_channels != out_channels * net_block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels * net_block.expansion, kernel_size=1, stride=stride, bias=False),
                      nn.BatchNorm2d(out_channels * net_block.expansion))

        layers = []
        layers.append(net_block(self.in_channels, out_channels, stride, downsample))
        if net_block.expansion != 1:
            self.in_channels = out_channels * net_block.expansion

        else:
            self.in_channels = out_channels

        for i in range(1, num_blocks):
            layers.append(net_block(self.in_channels, out_channels, 1, None))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpooling(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpooling(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def ResNet_n(num_layers):
    if num_layers == 18:
        # ResNet18
        model = ResNet(basic_block, [2, 2, 2, 2], num_classes)

    elif num_layers == 34:
        # ResNet34
        model = ResNet(basic_block, [3, 4, 6, 3], num_classes)

    elif num_layers == 50:
        # ResNet50
        model = ResNet(bottleneck_block, [3, 4, 6, 3], num_classes)

    elif num_layers == 101:
        # ResNet101
        model = ResNet(bottleneck_block, [3, 4, 23, 3], num_classes)

    elif num_layers == 152:
        # ResNet152
        model = ResNet(bottleneck_block, [3, 8, 36, 3], num_classes)

    else:
        print("error")

        return

    return model

class inceptionv1_block(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2_step1, out_channels2_step2, out_channels3_step1, out_channels3_step2, out_channels4):
        super(inceptionv1_block, self).__init__()
        self.branch1_conv = nn

class inceptionv1_block(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2_step1, out_channels2_step2, out_channels3_step1, out_channels3_step2, out_channels4):
        super(inceptionv1_block, self).__init__()
        self.branch1_conv = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels1, kernel_size=1),
                          nn.ReLU(inplace=True))
        
        self.branch2_conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels2_step1, kernel_size=1),
                          nn.ReLU(inplace=True))
        self.branch2_conv2 = nn.Sequential(nn.Conv2d(in_channels=out_channels2_step1, out_channels=out_channels2_step2, kernel_size=3, padding=1),
                          nn.ReLU(inplace=True))
        
        self.branch3_conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels3_step1, kernel_size=1),
                          nn.ReLU(inplace=True))
        self.branch3_conv2 = nn.Sequential(nn.Conv2d(in_channels=out_channels3_step1, out_channels=out_channels3_step2, kernel_size=5, padding=2),
                          nn.ReLU(inplace=True))
        
        self.branch4_maxpooling = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels4, kernel_size=1),
                          nn.ReLU(inplace=True))
     
    def forward(self, x):
        out1 = self.branch1_conv(x)
        out2 = self.branch2_conv2(self.branch2_conv1(x))
        out3 = self.branch3_conv2(self.branch3_conv1(x))
        out4 = self.branch4_conv1(self.branch4_maxpooling(x))
        out = torch.cat([out1, out2, out3, out4], dim=1)

        return out
      
class auxiliary_classifiers(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(auxiliary_classifiers, self).__init__()
        self.avgpooling = nn.AvgPool2d(kernel_size=5, stride=3)
        
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=1)
        
        self.fc1 = nn.Linear(in_features=128*4*4, out_features=1024)

        self.fc2 = nn.Linear(in_features=1024, out_features=out_channels)
     
    def forward(self, x):
        x = self.avgpooling(x)
        x = F.relu(self.conv(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5)
        x = self.fc2(x)

        return x
      
class InceptionV1(nn.Module):
    def __init__(self, num_classes, training=True):
        super(InceptionV1, self).__init__()
        self.training = training
        self.conv = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
                      nn.ReLU(inplace=True),
                      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1),
                      nn.ReLU(inplace=True),
                      nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
                      nn.ReLU(inplace=True),
                      nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        self.inception1 = inceptionv1_block(in_channels=192, out_channels1=64, out_channels2_step1=96, out_channels2_step2=128, out_channels3_step1=16, out_channels3_step2=32, out_channels4=32)
        self.inception2 = inceptionv1_block(in_channels=256, out_channels1=128, out_channels2_step1=128, out_channels2_step2=192, out_channels3_step1=32, out_channels3_step2=96, out_channels4=64)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception3 = inceptionv1_block(in_channels=480, out_channels1=192, out_channels2_step1=96, out_channels2_step2=208, out_channels3_step1=16, out_channels3_step2=48, out_channels4=64)

        if self.training == True:
            self.auxiliary1 = auxiliary_classifiers(in_channels=512,out_channels=num_classes)

        self.inception4 = inceptionv1_block(in_channels=512 ,out_channels1=160, out_channels2_step1=112, out_channels2_step2=224, out_channels3_step1=24, out_channels3_step2=64, out_channels4=64)
        self.inception5 = inceptionv1_block(in_channels=512, out_channels1=128, out_channels2_step1=128, out_channels2_step2=256, out_channels3_step1=24, out_channels3_step2=64, out_channels4=64)
        self.inception6 = inceptionv1_block(in_channels=512, out_channels1=112, out_channels2_step1=144, out_channels2_step2=288, out_channels3_step1=32, out_channels3_step2=64, out_channels4=64)

        if self.training == True:
            self.auxiliary2 = auxiliary_classifiers(in_channels=528,out_channels=num_classes)

        self.inception7 = inceptionv1_block(in_channels=528, out_channels1=256, out_channels2_step1=160, out_channels2_step2=320, out_channels3_step1=32, out_channels3_step2=128, out_channels4=128)
        self.maxpooling2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception8 = inceptionv1_block(in_channels=832, out_channels1=256, out_channels2_step1=160, out_channels2_step2=320, out_channels3_step1=32, out_channels3_step2=128, out_channels4=128)
        self.inception9 = inceptionv1_block(in_channels=832, out_channels1=384, out_channels2_step1=192, out_channels2_step2=384, out_channels3_step1=48, out_channels3_step2=128, out_channels4=128)

        self.avgpooling = nn.AvgPool2d(kernel_size=7,stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(in_features=1024,out_features=num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.maxpooling1(x)
        x = self.inception3(x)
        aux1 = self.auxiliary1(x)
        x = self.inception4(x)
        x = self.inception5(x)
        x = self.inception6(x)
        aux2 = self.auxiliary2(x)
        x = self.inception7(x)
        x = self.maxpooling2(x)
        x = self.inception8(x)
        x = self.inception9(x)
        x = self.avgpooling(x)
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1)
        out = self.fc(x)

        if self.training == True:
            return aux1, aux2, out

        else:
            return out
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
 
# Hyper-parameter
device = "cuda" if torch.cuda.is_available() else "cpu"
epoches = 300
patience = 20
#model = Classifier().to(device)
#model = InceptionV1(3, training=True).to(device)
model = ResNet(basic_block, [2, 2, 2, 2], 3).to(device)
loss_func = nn.CrossEntropyLoss()
batch_size = 64
lr = 1e-3
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

name = "github-version-14"

wandb.login(key = os.getenv('WANDB_API_KEY'))
wandb.init(
    project = "Breast-Cancer-Image-Diganosis",
    name = name,
    notes = "Full mammogram image only, ResNet, spliting train data to validate, random H flip",
    tags = ["baseline", "10.5555/2999134.2999257"],
    config = {
        "learning rate": lr,
        "architecture": "ResNet",
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

