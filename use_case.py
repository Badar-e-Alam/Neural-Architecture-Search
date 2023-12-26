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
from sklearn.metrics import roc_auc_score
import torchvision
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

from pytorchtool import EarlyStopping
# expriement 4 with dropout
# exprimets 5 without dropout expriment 6 is using the logits for loss calculation, expriement 7 with dropout and logits expriment 8 with cosine annealing and weight decay
writer = SummaryWriter("juma")

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
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = hw_api.get_net_config(selected_network[3], "cifar100")
single_network = get_cell_based_tiny_net(config)

data_flag = "pathmnist"
# data_flag = 'breastmnist'
download = True

NUM_EPOCHS = 10

BATCH_SIZE = 256
lr = 0.001

info = INFO[data_flag]
task = info["task"]

n_channels = info["n_channels"]
n_classes = len(info["label"])

DataClass = getattr(medmnist, info["python_class"])
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])


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
train_dataset = DataClass(split="train",
                          transform=data_transform,
                          download=download)
test_dataset = DataClass(split="test",
                         transform=data_transform,
                         download=download)

pil_dataset = DataClass(split="train", download=download, transform=None)

train_loader = data.DataLoader(dataset=train_dataset,
                               batch_size=BATCH_SIZE,
                               shuffle=True)
train_loader_at_eval = data.DataLoader(dataset=pil_dataset,
                                       batch_size=BATCH_SIZE,
                                       shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False)

print(train_dataset)
print("===================")
print(test_dataset)

# defining the model


class Net(nn.Module):
    def __init__(self,
                 model,
                 dropout_enabled=True,
                 n_classes=9,
                 dropout_rate=0.3):
        super(Net, self).__init__()
        self.model = model

        # Enable learning for all layers
        for param in self.model.parameters():
            param.requires_grad = True

        # Modify the classifier layer for the number of classes
        self.model.classifier = nn.Sequential(
            nn.ReLU(), nn.Linear(64, n_classes, bias=True))

    def forward(self, x):
        x = self.model(x)
        return x



# define loss function and optimizer


# def train(epoch, model, optimizer, criterion, train_loader, test_loader):
#     print(f"Training loop: {epoch}")
#     model.train()
#     for batch_idx, (data, target) in tqdm.tqdm(enumerate(train_loader)):
#         data, target = data.to(DEVICE), target.to(DEVICE)
#         optimizer.zero_grad()
#         logits, output = model(data)
#         loss = criterion(logits, target.squeeze().long())
#         loss.backward()
#         optimizer.step()
#         if batch_idx % 100 == 0:
#             print(
#                 f"Train Epoch: {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)} ({100.*batch_idx/len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
#             )
#             print("Learning rate: {}".format(optimizer.param_groups[0]["lr"]))

#     writer.add_scalar("learning rate", optimizer.param_groups[0]["lr"], epoch)
#     writer.add_scalar("training loss", loss.item(), epoch)
#     test_loss, auc, accuracy = test(epoch,
#                                     model,
#                                     criterion,
#                                     test_loader,
#                                     split="test")

#     if epoch % 5 == 0:
#         torch.save(model.state_dict(), "model.pth")

#     return test_loss, auc, accuracy


# def test(epoch, model, criterion, test_loader, split):
#     print(f"Testing loop: {epoch}")
#     model.eval()
#     test_loss = 0
#     correct = []

#     all_targets = []
#     all_predictions = []
#     all_prdicted_classes = []
#     correct_predictions = 0
#     with torch.no_grad():
#         for batch_idx, (data, target) in tqdm.tqdm(enumerate(test_loader)):
#             print(f"batch index: {batch_idx}")
#             print(f"batch data: {data.shape} and batch target: {target.shape}")
#             data, target = data.to(DEVICE), target.to(DEVICE)
#             logits, output = model(data)
#             import pdb; pdb.set_trace()
#             test_loss += criterion(output, target.squeeze().long()).item()
#             _, predicted_classes = torch.max(output, 1)

#             # Assuming target is a 2D tensor, squeeze it to make it 1D if needed
#             target = target.squeeze()

#             # Compare predictions with ground truth
#             correct_predictions += (predicted_classes == target).sum().item()

#             # Store targets and predictions for AUC calculation
#             all_targets.extend(target.cpu().numpy())
#             all_predictions.extend(output.cpu().numpy())
#             all_prdicted_classes.append(predicted_classes.cpu().numpy())
#         # Calculate accuracy after processing all batches

#         # Calculate AUC
#         # Binarize the output
#         import pdb
#         pdb.set_trace()
#         accuracy = correct_predictions / len(test_loader.dataset)
#         test_loss /= len(test_loader.dataset)
#         all_targets = label_binarize(all_targets,
#                                      classes=[0, 1, 2, 3, 4, 5, 6, 7, 8])
#         all_predictions = np.array(all_predictions)
#         auc = roc_auc_score(all_targets, all_predictions, multi_class="ovr")

#         # Add to Tensorboard
#         print(f"Test accuracy: {accuracy:.4f}")
#         print(f"Test loss: {test_loss:.4f}")
#         print(f"Test AUC: {auc:.4f}")
#         writer.add_scalar("Test auc", auc, epoch)
#         writer.add_scalar("Test loss", test_loss, epoch)
#         writer.add_scalar("Test accuracy", accuracy, epoch)
#         return test_loss, auc, accuracy


# learning_rate = 0.1  # the initial learning rate
# weight_decay = 0.0005  # the weight decay for regularization
# momentum = 0.9
# def test(split, model, epoch):
#     print(f"Testing loop: {epoch}")
#     model.eval()
#     y_true = torch.tensor([], dtype=torch.long, device=DEVICE)
#     y_score = torch.tensor([], device=DEVICE)

#     # Evaluation loop
#     model.eval()
#     with torch.no_grad():
#         for inputs, targets in tqdm.tqdm(test_loader):
#             inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
#             outputs = model(inputs)

#             y_true = torch.cat((y_true, targets), 0)
#             y_score = torch.cat((y_score, outputs[1]), 0)

#     # Move tensors to CPU for sklearn compatibility
#     y_true = y_true.cpu().numpy()
#     y_score = y_score.cpu().numpy()

#     # Calculate accuracy
#     accuracy = (y_score.argmax(axis=1) == y_true).mean()
#     print(f'Accuracy: {accuracy * 100:.2f}%')

#     # Binarize the output for multi-class AUC calculation
#     y_true_bin = label_binarize(y_true, classes=list(range(9)))
#     y_score_bin = y_score

#     # Calculate AUC for each class
#     aucs = [roc_auc_score(y_true_bin[:, i], y_score_bin[:, i]) for i in range(9)]

#     # Calculate average AUC
#     avg_auc = sum(aucs) / len(aucs)
#     print(f'Average AUC: {avg_auc:.2f}')
#     writer.add_scalar('test auc',avg_auc, epoch)
#     writer.add_scalar('test acc', accuracy, epoch)


def main():

    model = Net(single_network, dropout_enabled=True, n_classes=n_classes)
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters())
    train(model, train_loader, test_loader, optimizer, criterion, writer)

    # resnet = torchvision.models.resnet50()
    # # Count parameters for resnet
    # resnet_params = sum(p.numel() for p in resnet.parameters())
    # print(f"Number of parameters in resnet: {resnet_params}")

    # # Count parameters for model
    # model_params = sum(p.numel() for p in model.parameters())
    # print(f"Number of parameters in model: {model_params}")
    # resnet_percentage = (model_params / (resnet_params + model_params)) * 100
    # print(
    #     f"Percentage of model parameters from ResNet: {resnet_percentage:.2f}%"
    # )

    # if task == "multi-label, binary-class":
    #     criterion = nn.BCEWithLogitsLoss()
    # else:
  

    # # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
    # #                                                        T_max=200)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(f"couda is available: {device}")
    # print(f"training is starting")
    # patience = 20
    # early_stopping = EarlyStopping(patience=patience, verbose=True)
    # # for epoch in range(NUM_EPOCHS):

    # #     test_loss, auc, acc = train(epoch, model, optimizer, criterion,
    # #                                 train_loader, test_loader)
    # #     # #test_loss, _, _= test(epoch,
    # #     #                        model,
    # #     #                        criterion,
    # #     #                        test_loader,
    # #     #                        split="test")
    # #     early_stopping(test_loss, model)
    # #     scheduler.step()
    # #     # if optimizer.param_groups[0]["lr"] > lower_bound:
    # #     scheduler.step()
    # for epoch in range(NUM_EPOCHS):
    #     train_correct = 0
    #     train_total = 0
    #     test_correct = 0
    #     test_total = 0

    #     model.train()
    #     for index, (inputs, targets) in tqdm.tqdm(enumerate(train_loader)):
    #         # forward + backward + optimize
    #         optimizer.zero_grad()
    #         outputs = model(inputs.to(DEVICE))
    #         targets = targets.to(DEVICE)
    #         if task == 'multi-label, binary-class':
    #             targets = targets.to(torch.float32)
    #             loss = criterion(outputs[0], targets.squeeze().long())
    #         else:
    #             targets = targets.squeeze().long()
    #             loss = criterion(outputs[0], targets)

    #         loss.backward()
    #         optimizer.step()

    #         if index % 200 == 0:
    #             print('[%d, %5d] loss: %.3f' %
    #                   (epoch + 1, index + 1, loss.item()))
    #         # if epoch % 4 == 0:
    #         #     torch.save(model.state_dict(), 'model.pth')
    #         #     test('test', model, epoch)

    #         writer.add_scalar('training loss', loss.item(), epoch)
    #     # evaluation


    # # print('==> Evaluating ...')
    # # test('train')
    # test('test',model,epoch)

def evaluate(net, dataloader, mode, trial_num, criterion):
    print("evaluating...", trial_num)
    net.eval()
    test_loss = 0
    correct = 0

    total_samples = len(test_loader.dataset)

    with torch.no_grad():
        for batch_idx, (data, target) in tqdm.tqdm(enumerate(dataloader)):
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = net(data)
            test_loss += criterion(output[1], target.squeeze().long()).item()
            _, predicted_classes = torch.max(output[1], 1)

            # Assuming target is a 2D tensor, squeeze it to make it 1D if needed
            target = target.squeeze()

            # Compare predictions with ground truth
            correct_predictions = (predicted_classes == target).sum().item()
            correct += correct_predictions
            import pdb ; pdb.set_trace()
        # Calculate accuracy after processing all batches

        accuracy = correct / total_samples


        print(f"Total Accuracy: {accuracy * 100:.2f}%")

    return accuracy,test_loss/total_samples


def train(net, trainloader, validloader, optimizer, criterion, writer):
    best_valid_accuracy = 0.0

    for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize

            outputs = net(inputs.to(DEVICE))
            train_loss = criterion(outputs[1], labels.squeeze().long().to(DEVICE))
            train_loss.backward()
            optimizer.step()


            if i % 100 == 0:
                print("train_loss: {}".format(train_loss.item()))


        # for i, data in enumerate(validloader, 0):
        #     inputs, labels = data
        #     outputs = net(inputs.to(device))
        #     validation_loss = criterion(outputs, labels.squeeze().long().to(device))

        valid_accuracy,validation_loss = evaluate(net, validloader, "valid", epoch, criterion)

        # save best trained paramsclear
        # print("save best net at trial_num {}".format(trial_num))
        # torch.save(
        #         net.state_dict(),
        #         args.output_path + "/best_net_{:03}".format(trial_num) + ".pth.tar",
        # )

        print(
            "[{:03d} / {:03d}] train_loss, valid_loss, valid_accuracy : {:.3f}, {:.3f}, {:.3f}".format(
                epoch + 1, epoch, train_loss, validation_loss, valid_accuracy
            )
        )

        # if args.tensorboard:
            # writer.add_scalar('Loss/train', train_loss.avg, epoch)
        writer.add_scalars(
                "Loss/train", {"trial_{:03d}".format(epoch): train_loss}, epoch
            )
        writer.add_scalars(
                "Loss/valid", {"trial_{:03d}".format(epoch): validation_loss}, epoch
            )
        writer.add_scalars(
                "Accuracy/valid",
                {"trial_{:03d}".format(epoch): valid_accuracy},
                epoch,
            )

        writer.flush()

    return valid_accuracy
if __name__ == "__main__":
    main()
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
