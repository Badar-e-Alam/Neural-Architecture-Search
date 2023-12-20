from hw_nas_bench_api import HWNASBenchAPI as HWAPI
import json
import tqdm
from hw_nas_bench_api.nas_201_models import get_cell_based_tiny_net
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO, Evaluator
from torch.utils.tensorboard import SummaryWriter

# expriement 4 with dropout
# exprimets 5 without dropout expriment 6 is using the logits for loss calculation, expriement 7 with dropout and logits expriment 8 with cosine annealing and weight decay
writer = SummaryWriter("runs/medmnist_experiment_8")

hw_api = HWAPI("HW-NAS-Bench-v1_0.pickle", search_space="nasbench201")
with open("seleted_network.json", "r") as jsfile:
    selected_network = json.load(jsfile)


# for idx in tqdm.tqdm(selected_network):
#         print(f"Network index: {idx}")
#         for dataset in ["cifar10"]:
#             # HW_metrics = hw_api.query_by_index(idx, dataset)
# # engery_list.append(HW_metrics["fpga_energy"])
# # latency_list.append(HW_metrics["fpga_latency"])
# if HW_metrics["fpga_energy"]>7.6674888908800005:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = hw_api.get_net_config(selected_network[-2], "cifar10")
single_network = get_cell_based_tiny_net(config)


##loading dataset
class EarlyStopping:
    def __init__(self, patience=2, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.stop = False

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_score = score
            self.counter = 0


data_flag = "pathmnist"
# data_flag = 'breastmnist'
download = True

NUM_EPOCHS = 200

BATCH_SIZE = 256
lr = 0.001

info = INFO[data_flag]
task = info["task"]

n_channels = info["n_channels"]
n_classes = len(info["label"])

DataClass = getattr(medmnist, info["python_class"])
data_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(28),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)


# train_transform = transforms.Compose(
#     [
#         transforms.RandomResizedCrop(
#             28
#         ),  # Randomly crop the image and resize it to 224x224
#         transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
#         transforms.ToTensor(),  # Convert the image to a PyTorch tensor
#         transforms.Normalize(
#             mean=[0.5], std=[0.5]
#         ),  # Normalize the pixel values to the range [0, 1]
#     ]
# )

# # Define data transformations for validation/testing
# val_transform = transforms.Compose(
#     [
#         transforms.Resize(28),  # Resize the image to 256x256
#         transforms.CenterCrop(28),  # Crop the center of the image to 224x224
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5], std=[0.5]),
#     ]
# )

# load the data
train_dataset = DataClass(split="train", transform=data_transform, download=download)
test_dataset = DataClass(split="test", transform=data_transform, download=download)

pil_dataset = DataClass(split="train", download=download, transform=None)

train_loader = data.DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
train_loader_at_eval = data.DataLoader(
    dataset=pil_dataset, batch_size=BATCH_SIZE, shuffle=False
)
test_loader = data.DataLoader(
    dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False
)


print(train_dataset)
print("===================")
print(test_dataset)

# defining the model


class Net(nn.Module):
    def __init__(self, model, dropout_enabled=True, n_classes=9):
        super(Net, self).__init__()
        self.model = model
        self.model.classifier = nn.Linear(64, n_classes)
        self.dropout_enabled = dropout_enabled
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.model(x)
        if self.dropout_enabled:
            x = self.dropout(x[1])
        return x


# define loss function and optimizer


def train(epoch, model, optimizer, criterion, train_loader):
    print(f"Training loop: {epoch}")
    model.train()
    for batch_idx, (data, target) in tqdm.tqdm(enumerate(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logits, output = model(data)
        loss = criterion(logits, target.squeeze().long())
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)} ({100.*batch_idx/len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )
            print("Learning rate: {}".format(optimizer.param_groups[0]["lr"]))
            writer.add_scalar(
                "learning rate",
                optimizer.param_groups[0]["lr"],
                epoch,
            )
            writer.add_scalar(
                "training loss",
                loss.item(),
                epoch,
            )
    if epoch % 5 == 0:
        torch.save(model.state_dict(), "model.pth")


def test(epoch, model, criterion, test_loader, split):
    print(f"Testing loop: {epoch}")
    model.eval()
    test_loss = 0
    correct = 0
    # if split == 'train':
    #     total_samples = len(train_loader_at_eval)
    #     test_loader = train_loader_at_eval
    # else:
    total_samples = len(test_loader.dataset)

    with torch.no_grad():
        for batch_idx, (data, target) in tqdm.tqdm(enumerate(test_loader)):
            data, target = data.to(device), target.to(device)
            logits, output = model(data)
            test_loss += criterion(logits, target.squeeze().long()).item()
            _, predicted_classes = torch.max(logits, 1)

            # Assuming target is a 2D tensor, squeeze it to make it 1D if needed
            target = target.squeeze()

            # Compare predictions with ground truth
            correct_predictions = (predicted_classes == target).sum().item()
            correct += correct_predictions

        # Calculate accuracy after processing all batches

        accuracy = correct / total_samples

        print(f"Total Accuracy: {accuracy * 100:.2f}%")
        # EarlyStopping(accuracy*100)
        # Calculate average test loss
        test_loss /= total_samples
        print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy {accuracy}\n")

        # Add to Tensorboard
        writer.add_scalar("test loss", test_loss, epoch)
        writer.add_scalar("test accuracy", 100.0 * accuracy, epoch)
        return test_loss, accuracy


learning_rate = 0.1  # the initial learning rate
weight_decay = 0.0005  # the weight decay for regularization
momentum = 0.9
model = Net(single_network, dropout_enabled=False, n_classes=n_classes)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"couda is available: {device}")
print(f"training is starting")
for epoch in range(NUM_EPOCHS):
    train(epoch, model, optimizer, criterion, train_loader)
    test_loss, _ = test(epoch, model, criterion, test_loader, split="test")
    scheduler.step()
    # scheduler.step()
    # early_stopping(test_loss)
    # if early_stopping.stop:
    #     print("Early stopping")
    #     torch.save(model.state_dict(), 'model.pth')
    #     break
# for epoch in range(NUM_EPOCHS):
#     train_correct = 0
#     train_total = 0
#     test_correct = 0
#     test_total = 0

#     model.train()
#     for index,(inputs, targets) in tqdm.tqdm(enumerate(train_loader)):
#         # forward + backward + optimize
#         optimizer.zero_grad()
#         outputs = model(inputs)

#         if task == 'multi-label, binary-class':
#             targets = targets.to(torch.float32)
#             loss = criterion(outputs, targets)
#         else:
#             targets = targets.squeeze().long()
#             loss = criterion(outputs[1], targets)
#         loss.backward()
#         optimizer.step()

#         if index % 100 == 0:
#             print('[%d, %5d] loss: %.3f' % (epoch + 1, index + 1, loss.item()))
#     torch.save(model.state_dict(), 'model.pth')
#     writer.add_scalar('training loss', loss.item(), epoch)


# # evaluation

# def test(split):
#     model.eval()
#     y_true = torch.tensor([])
#     y_score = torch.tensor([])

#     data_loader = train_loader_at_eval if split == 'train' else test_loader

#     with torch.no_grad():
#         for inputs, targets in data_loader:
#             outputs = model(inputs)

#             if task == 'multi-label, binary-class':
#                 targets = targets.to(torch.float32)
#                 outputs = outputs[1].softmax(dim=-1)
#             else:
#                 targets = targets.squeeze().long()
#                 outputs = outputs[1].softmax(dim=-1)
#                 targets = targets.float().resize_(len(targets), 1)

#             y_true = torch.cat((y_true, targets), 0)
#             y_score = torch.cat((y_score, outputs), 0)

#         y_true = y_true.numpy()
#         y_score = y_score.detach().numpy()
#         import pdb; pdb.set_trace()
#         evaluator = Evaluator(data_flag, split)
#         metrics = evaluator.evaluate(y_score)

#         print('%s  auc: %.3f  acc:%.3f' % (split, *metrics))


# print('==> Evaluating ...')
# test('train')
# test('test')
