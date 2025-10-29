from pyexpat import model
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
import torchvision.transforms.v2 as randomresize
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn.init as init
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchinfo
from torchvision.models import vgg16, VGG16_Weights

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class VGG_19(nn.Module):
    def __init__(self, num_classes = 1000, init_weights = True) -> None:
        super(VGG_19, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),#추가
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),#추가
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),#추가
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # self.classifier = nn.Sequential(
        #     nn.Linear(512*7*7, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(0.5),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(0.5),
        #     nn.Linear(4096, num_classes)
        # )
        
        #밀집 평가(Dense Evaluation) 방식
        self.denses = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=7, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=4096, out_channels=1000, kernel_size=1, padding=1, stride=1),
            nn.AdaptiveAvgPool2d((1,1))
        )
        
        if init_weights:
                self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight) # xavier 초기화
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)



    def forward(self, x):
        if isinstance(x, list):
            outputs = []            
            for t in x:
                out= self.features(t)
                out = self.avgpool(out)
                out = self.denses(out)
                out = torch.flatten(out, 1)
                outputs.append(out)
            final_output = torch.stack(outputs).mean(dim=0)
            return final_output
            
        else:
            x = self.features(x)
            x = self.avgpool(x)
            x = self.denses(x)
            x = torch.flatten(x, 1)
            # x = self.classifier(x)
            return x


def train_model(model, train_dataloader, val_dataloader, criterion, scheduler, optimizer, num_epochs) :
    model.to(device)
    torch.backends.cudnn.benchmark = True

    train_accuracy_list = []
    val_accuracy_list = []
    train_loss_list = []
    val_loss_list = []

    for epoch in range(num_epochs) :
        print(f'Epoch {epoch + 1}/ {num_epochs}')
        print('*' * 30)

        # ====== 학습(Training) 단계 ======
        model.train() # 모델을 학습 모드로 설정

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_dataloader, desc="Training"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels)

        epoch_train_loss = running_loss / len(train_dataloader)
        epoch_train_acc = running_corrects.double() / len(train_dataloader.dataset)

        print(f'Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc.double():.4f}')

        # ====== 검증(Validation) 단계 ======
        model.eval() # 모델을 평가 모드로 설정

        best_val_acc = 0.0
        val_running_loss = 0.0
        val_running_corrects = 0

        with torch.no_grad(): # 검증 단계에서는 그라디언트 계산 비활성화
            for inputs, labels in tqdm(val_dataloader, desc="Validation"):
                inputs = [t.to(device) for t in inputs]
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_running_loss += loss.item()
                val_running_corrects += torch.sum(preds == labels)

        epoch_val_loss = val_running_loss / len(val_dataloader.dataset)
        epoch_val_acc = val_running_corrects.double() / len(val_dataloader.dataset)
        scheduler.step(epoch_val_acc)

        print(f'Validation Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc.double():.4f}')
        print('*' * 30)

        train_accuracy_list.append(epoch_train_acc.item())
        train_loss_list.append(epoch_train_loss)
        val_accuracy_list.append(epoch_val_acc.item())
        val_loss_list.append(epoch_val_loss)

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            print(f'New best model found at epoch {epoch + 1} with Validation Accuracy: {best_val_acc:.4f}. Saving checkpoint...')
            torch.save(model.state_dict(), 'best_vgg19_checkpoint1.pth')

    return train_accuracy_list, val_accuracy_list, train_loss_list, val_loss_list

def test_model(net, dataloader, criterion, num_epochs):
    net.to(device)
    net.eval()
    accuracy_list = []
    loss_list = []

    for epoch in range(num_epochs) :
        print(f'Epoch {epoch + 1}/ {num_epochs}')
        print('*' * 30)

    epoch_test_loss = 0.0
    epoch_test_corrects = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = [t.to(device) for t in inputs]
            labels = labels.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            epoch_test_loss += loss.item()
            epoch_test_corrects += torch.sum(preds == labels.data)

        epoch_test_loss = epoch_test_loss / len(dataloader.dataset)
        epoch_acc = epoch_test_corrects.double() / len(dataloader.dataset)

        print(f'Loss: {epoch_test_loss:.4f} Acc: {epoch_acc:.4f}')

        accuracy_list.append(epoch_acc.item())
        loss_list.append(epoch_test_loss)
    return accuracy_list, loss_list

#다중 스케일(multiple scale)
#train (fine-tuning with vgg19)

torch.manual_seed(56)
np.random.seed(56)
random.seed(56)

Q = [224, 256, 288]
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

def custom_collate_fn(batch):    
    # 각 스케일(변환)별로 텐서 리스트를 분리 (Transpose)
    num_scales = len(batch[0][0])
    scaled_inputs = [[] for _ in range(num_scales)]
    
    for item in batch:
        for i in range(num_scales):
            scaled_inputs[i].append(item[0][i])
            
    final_inputs = [torch.stack(tensors) for tensors in scaled_inputs]

    labels = [item[1] for item in batch]
    final_labels = torch.as_tensor(labels)

    return final_inputs, final_labels

# 기본 전처리 (Resize + ToTensor + Normalize)
def get_test_transform(size):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

# 하나의 이미지를 여러 버전으로 변환하는 transform
class MultiScaleTransform:
    def __init__(self, sizes):
        self.transforms = [get_test_transform(s) for s in sizes]

    def __call__(self, img):
        return [t(img) for t in self.transforms]
#(3, H, W), ... list

# === 데이터셋 준비 ===
# Train transform 고정된 s 256
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop((224, 224), scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
test_transform = MultiScaleTransform(Q)

train_dataset = datasets.CIFAR10(root='../dataset', train=True, download=True, transform=train_transform)
valid_dataset = datasets.CIFAR10(root='../dataset', train=False, download=True, transform=test_transform)
test_dataset = datasets.CIFAR10(root='../dataset', train=False, download=True, transform=test_transform)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)

model = VGG_19(num_classes=1000, init_weights=True)
pre_trained_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
# print(pre_trained_model)
pre_trained_weights = pre_trained_model.state_dict()
model_state_dict = model.state_dict()

from collections import OrderedDict

pre_trained_weights = pre_trained_model.state_dict()
target_state_dict = model.state_dict()

remap_dict = OrderedDict()

# 첫 4개의 features layer만 가져오기
for k, v in pre_trained_weights.items():
    if k.startswith('features.'):
        layer_idx = int(k.split('.')[1])
        if layer_idx < 4:  # 0~3 layer만 복사
            remap_dict[k] = v

# 매핑된 가중치만 적용
new_weights = {}
for k_custom, v_custom in target_state_dict.items():
    if k_custom in remap_dict and v_custom.shape == remap_dict[k_custom].shape:
        new_weights[k_custom] = remap_dict[k_custom]

model_state_dict.update(new_weights)

# FC layer 가중치를 Conv layer로 변환하여 복사
fc_layers = [pre_trained_model.classifier[0], pre_trained_model.classifier[3], pre_trained_model.classifier[6]]
conv_layers = [model.denses[0], model.denses[2], model.denses[4]]

for fc, conv in zip(fc_layers, conv_layers):
    conv.weight.data.copy_(fc.weight.data.view(conv.weight.shape))
    conv.bias.data.copy_(fc.bias.data)
    
model.load_state_dict(model_state_dict)

# 마지막 레이어 교체
num_ftrs = model.denses[4].in_channels
# model.classifier[6] = nn.Linear(num_ftrs, 10)
model.denses[4] = nn.Conv2d(in_channels=num_ftrs, out_channels=10, kernel_size=1, padding=1, stride=1)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9, weight_decay=5e-4)
#val_loss가 향상되지 않으면 lr을 1/10 줄임
scheduler = ReduceLROnPlateau(optimizer, 'max', patience=1, factor=0.1)
num_epochs =74#원래는 74
train_accuracy_list, val_accuracy_list, train_loss_list, val_loss_list = train_model(model, train_dataloader, valid_dataloader, criterion, scheduler, optimizer, num_epochs =num_epochs)

# 그래프 그리기
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor="w")

# 정확도(Accuracy) 그래프
ax1.plot(train_accuracy_list, label="Train Accuracy")
ax1.plot(val_accuracy_list, label="Validation Accuracy")
ax1.set_title("Accuracy over Epochs")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.legend()
ax1.grid(True)

# 손실(Loss) 그래프
ax2.plot(train_loss_list, label="Train Loss")
ax2.plot(val_loss_list, label="Validation Loss")
ax2.set_title("Loss over Epochs")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('vgg19_train_val_loss_plot1.png')
plt.show()
#torchinfo.summary(model, input_size=(1, 3, 224, 224))
test_acc, test_loss = test_model(model, test_dataloader, criterion, num_epochs=1)

