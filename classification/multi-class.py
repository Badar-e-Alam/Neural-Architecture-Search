import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import tqdm
import medmnist
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from medmnist import INFO, Evaluator
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import sys

sys.path.append("..")  # Add parent folder to the sys.path
from hw_nas_bench_api import HWNASBenchAPI as HWAPI
import json
from hw_nas_bench_api.nas_201_models import get_cell_based_tiny_net

# tensorboard --logdir=runs
writer = SummaryWriter("runs/multi-class")

# Hyperparameters
batch_size = 256
num_epochs = 100
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make_model(path, num_classes):
    hw_api = HWAPI(path, search_space="nasbench201")
    with open("seleted_network.json", "r") as f:
        network = json.load(f)

    max_params = 0
    selected_model = None

    for index, net in enumerate(network):
        config = hw_api.get_net_config(net, "cifar10")
        single_network = get_cell_based_tiny_net(config)
        model = Net(single_network, n_classes=num_classes)
        model = model.to(DEVICE)

        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model {index} parameters:", num_params)
        if num_params > max_params:
            max_params = num_params
            selected_model = model
    print("Selected model with max parameters: ", max_params)
    return selected_model


class Net(nn.Module):
    def __init__(self, model, dropout_enabled=True, n_classes=9, dropout_rate=0.3):
        super(Net, self).__init__()
        self.model = model

        # Enable learning for all layers
        for param in self.model.parameters():
            param.requires_grad = True

        # Modify the classifier layer for the number of classes
        self.model.classifier = nn.Sequential(
            # nn.Dropout(p=dropout_rate), #added dropout layer because model was overfitting
            nn.ReLU(),
            nn.Linear(64, n_classes, bias=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x



def get_data():
    data_flag = "pathmnist"
    info = INFO[data_flag]
    task = info["task"]
    n_channels = info["n_channels"]
    n_classes = len(info["label"])
    DataClass = getattr(medmnist, info["python_class"])
    data_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
    )
    train_dataset = DataClass(split="train", transform=data_transform, download=True)
    test_dataset = DataClass(split="test", transform=data_transform, download=True)

    pil_dataset = DataClass(split="train", download=True, transform=None)

    train_loader = data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    train_loader_at_eval = data.DataLoader(
        dataset=pil_dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader, train_loader_at_eval, n_classes


def evaluate(model, dataloader, mode, trial_num, criterion):
    print("evaluating...", trial_num)
    model.eval()
    test_loss = 0
    total_correct = 0
    all_predictions = []
    true_label = []

    total_samples = 0

    with torch.no_grad():
        for batch_idx, (data, target) in tqdm.tqdm(enumerate(dataloader)):
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            test_loss += criterion(output[0], target.squeeze().long()).item()
            _, predicted_classes = torch.max(output[1], 1)  #
            output = F.softmax(output[1], dim=1)
            all_predictions.append(output.cpu().numpy())
            true_label.append(target.squeeze().cpu().numpy())

            # Assuming target is a 2D tensor, squeeze it to make it 1D if needed
            # target = target.squeeze()

            # Compare predictions with ground truth
            target = target.squeeze()
            correct_predictions = (predicted_classes == target).sum().item()
            total_correct += correct_predictions
            total_samples += target.shape[0]

            # Calculate accuracy after processing all batches
            # if batch_idx % 10 == 0:
            #     # print(
            #     #     f"Batch {batch_idx} of {len(dataloader)}: "
            #     #     f"Accuracy: {correct_predictions * 100 / target.shape[0]:.2f}%",
            #     # )

        y_true = label_binarize(np.concatenate(true_label, axis=0),
                                classes=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        all_predictions = np.concatenate(all_predictions, axis=0)
        auc = roc_auc_score(y_true, all_predictions, multi_class="ovr")
        accuracy = (total_correct / total_samples) * 100
        writer.add_scalar("accuracy", accuracy, trial_num)
        writer.add_scalar("auc", auc, trial_num)
        writer.add_scalar("loss", test_loss / total_samples, trial_num)

        print(f"Total Accuracy: {accuracy:.2f}%")
        print(f"Total AUC: {auc:.2f}%")
        print(f"Total Loss: {test_loss / total_samples:.2f}%")

    return accuracy, test_loss / total_samples, auc


def train(model, device, criterion, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output[0], target.squeeze().long())
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % 200 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )
    writer.add_scalar("Training loss",train_loss/(batch_idx*batch_size),epoch)
    return train_loss // (batch_idx + 1)


def main():
    train_loader, test_loader, train_loader_at_eval, n_classes = get_data()
    model = make_model("HW-NAS-Bench-v1_0.pickle", n_classes)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    schdeduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    resnet = torchvision.models.resnet50()
    # Count parameters for resnet
    resnet_params = sum(p.numel() for p in resnet.parameters())
    print(f"Number of parameters in resnet: {resnet_params}")

    # Count parameters for model
    model_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in model: {model_params}")
    resnet_percentage = (model_params / (resnet_params + model_params)) * 100
    print(
        f"Percentage of model parameters from ResNet: {resnet_percentage:.2f}%"
    )

    for epoch in (range(1, num_epochs + 1)):
        train(model, DEVICE,criterion, train_loader, optimizer, epoch)
        if epoch %10==0:
            accuracy, loss, auc = evaluate(model, test_loader, "train", epoch,
                                       criterion)
        writer.add_scalar("Learning_rate",optimizer.param_groups[0]["lr"])
        schdeduler.step()

    writer.close()


if __name__ == "__main__":
    main()
