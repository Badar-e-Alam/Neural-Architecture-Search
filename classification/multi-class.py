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
writer = SummaryWriter("classification/runs/multi-class")


def make_model(path, num_classes):
    hw_api = HWAPI(path, search_space="nasbench201")
    with open("seleted_network.json", "r") as f:
        network = json.load(f)
    config = hw_api.get_net_config(network[-2], "cifar10")
    single_network = get_cell_based_tiny_net(config)
    model = Net(single_network, n_classes=num_classes)
    model = model.to(DEVICE)
    return model


class Net(nn.Module):
    def __init__(self, model, dropout_enabled=True, n_classes=9, dropout_rate=0.3):
        super(Net, self).__init__()
        self.model = model

        # Enable learning for all layers
        for param in self.model.parameters():
            param.requires_grad = True


        # Modify the classifier layer for the number of classes
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.ReLU(), nn.Linear(64, n_classes, bias=True)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# Hyperparameters
batch_size = 128
num_epochs = 100
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
            test_loss += criterion(output[1], target.squeeze().long()).item()
            _, predicted_classes = torch.max(output[1], 1)#
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

        

        y_true = label_binarize(np.concatenate(true_label,axis=0), classes=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        all_predictions = np.concatenate(all_predictions,axis=0)
        auc = roc_auc_score(y_true, all_predictions, multi_class="ovr")
        accuracy = total_correct / total_samples
        # writer.add_scarlar('accuracy',accuracy,trial_num)
        # writer.add_scarlar('auc',auc,trial_num)
        # writer.add_scarlar('loss',test_loss/total_samples,trial_num)

        print(f"Total Accuracy: {accuracy * 100:.2f}%")

    return accuracy*100, test_loss / total_samples, auc


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output[0], target.squeeze().long())
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        # if batch_idx %200 == 0:
        #     print(
        #         "Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}".format(
        #             epoch,
        #             batch_idx * len(data),
        #             len(train_loader.dataset),
        #             100.0 * batch_idx / len(train_loader),
        #             loss.item(),
        #         )
        #     )
        #     break
    return train_loss//(batch_idx+batch_size)

def main():
    train_loader, test_loader, train_loader_at_eval, n_classes = get_data()
    model = make_model("HW-NAS-Bench-v1_0.pickle", n_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    print("Total Model parameters:", sum(p.numel() for p in model.parameters()))
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    print("Model Parameters:")
    # for name ,param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name,param.data.cpu().numpy())
    initial_params = {name: param.clone().detach().cpu().numpy() for name, param in model.named_parameters() if param.requires_grad}
    initial_all = np.concatenate([initial_params[name].flatten() for name in initial_params])

    prev_params = None
    
        
    for epoch in range(1, num_epochs + 1):
        train_loss=train(model, DEVICE, train_loader, optimizer, epoch)
        current_params = {name: param.clone().detach().cpu().numpy() for name, param in model.named_parameters() if param.requires_grad}
        print("ephch:",epoch)
        if prev_params is not None and epoch%10==0:
            prev_all = np.concatenate([prev_params[name].flatten() for name in prev_params])
            current_all = np.concatenate([current_params[name].flatten() for name in current_params])

            fig, axs = plt.subplots(2, figsize=(12, 14))

            axs[0].set_title("Previous Epoch Parameters")
            axs[0].plot(prev_all, label="Previous Epoch")
            axs[0].legend()

            axs[1].set_title("Current Epoch Parameters")
            axs[1].plot(current_all, label="Current Epoch")
            axs[1].legend()

            plt.show()
        elif epoch%4==0:
            print("saving the last model parameters")
            prev_params = current_params
        import pdb; pdb.set_trace()
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.data.cpu().numpy())
        accuracy, loss, auc = evaluate(model, test_loader, "train", epoch, criterion)
        writer.add_scalar("training loss", train_loss, epoch)
        writer.add_scalar("training accuracy", accuracy, epoch)
        writer.add_scalar("Test loss", loss, epoch)
        # accuracy,loss=evaluate(model, test_loader, "test", epoch, criterion)
        writer.add_scalar("testing AUC", auc, epoch)
    # writer.add_scalar('testing loss',
    #                     loss,
    #                     epoch)
        writer.close()


if __name__ == "__main__":
    main()
