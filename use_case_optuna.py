import json
import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import argparse
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO, Evaluator
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import optuna
from utils.model import Net

from utils.utils import str2bool, AverageMeter
from utils.utils import str2bool, AverageMeter

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



best_accuracy = 0.0

parser = argparse.ArgumentParser(description="Hyperparameter tuning")
parser.add_argument("-b", "--batch_size", type=int, default=256, help="batch size")
parser.add_argument(
    "-j", "--num_workers", type=int, default=4, help="number of workers"
)
parser.add_argument("-e", "--epochs", type=int, default=50, help="number of epochs")
parser.add_argument("-s", "--show_image", default=False, help="show image")
parser.add_argument("-p", "--output_path", default="./results", help="output path")
parser.add_argument(
    "-t",
    "--tensorboard",
    default=True,
    type=str2bool,
    help="tensorboard dump the logs for the training and validation",
)
parser.add_argument(
    "-o", "--optuna", default=True, type=str2bool, help="optuna for hypermeter tunning"
)
parser.add_argument(
    "--optuna_trialnum", default=100, type=int, help="optuna trial number(defualt: 20)"
)
args = parser.parse_args()
trial_num = -1
best_acc = 0.0  #
best_num = -1


def checkimages(loader, writer, mode="train"):
    dataiter = iter(loader)
    images, labels = next(dataiter)

    if args.show_image:
        imshow(torchvision.utils.make_grid(images))

    # print(' '.join('%5s' % classes[labels[j].item()] for j in range(images.size(0))))

    if args.tensorboard:
        grid = torchvision.utils.make_grid(images)
        writer.add_image("Images/{}".format(mode), grid / 2 + 0.5, 0)

        if mode == "train":
            net = Net().to(device)
            writer.add_graph(net, images.to(device))

    del dataiter


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def objective_variable(train_loader, test_loader, writer):
    def objective(trial):
        global trial_num
        trial_num += 1
        model = Net().to(device)
        optimizer = get_optimizer(trial, model)
        criterion = nn.CrossEntropyLoss()
        validation_acc = train(
            model, train_loader, test_loader, optimizer, criterion, writer, trial_num
        )
        return validation_acc

    return objective


def get_optimizer(trial, model):
    optimizer_names = ["Adam", "SGD", "RMSprop"]
    optimizer_name = trial.suggest_categorical("optimizer", optimizer_names)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-10, 1e-3)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    optimizer = None
    if optimizer_name == "Adam":
        adam_betas = trial.suggest_categorical(
            "adam_betas", [(0.9, 0.999), (0.8, 0.999), (0.7, 0.999), (0.6, 0.999)]
        )
        optimizer = optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay, betas=adam_betas
        )
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
        )
    else:
        optimizer = optim.RMSprop(
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
        )
    return optimizer


def evaluate(net, dataloader, mode, trial_num, criterion):
    print("evaluating...", trial_num)
    net.eval()
    test_loss = 0
    correct = 0

    total_samples = len(test_loader.dataset)

    with torch.no_grad():
        for batch_idx, (data, target) in tqdm.tqdm(enumerate(dataloader)):
            data, target = data.to(device), target.to(device)
            output = net(data)
            test_loss += criterion(output, target.squeeze().long()).item()
            _, predicted_classes = torch.max(output, 1)

            # Assuming target is a 2D tensor, squeeze it to make it 1D if needed
            target = target.squeeze()

            # Compare predictions with ground truth
            correct_predictions = (predicted_classes == target).sum().item()
            correct += correct_predictions
        # Calculate accuracy after processing all batches

        accuracy = correct / total_samples


        print(f"Total Accuracy: {accuracy * 100:.2f}%")

    return accuracy,test_loss/total_samples


def train(net, trainloader, validloader, optimizer, criterion, writer, trial_num):
    best_valid_accuracy = 0.0
    print("trial id : {}".format(trial_num))

    for epoch in range(args.epochs):  # loop over the dataset multiple times

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize

            outputs = net(inputs.to(device))
            train_loss = criterion(outputs, labels.squeeze().long().to(device))
            train_loss.backward()
            optimizer.step()


            if i % 100 == 0:
                print("train_loss: {}".format(train_loss.item()))

        torch.save(net.state_dict(), args.output_path + "/checkpoint.pth.tar")

        # for i, data in enumerate(validloader, 0):
        #     inputs, labels = data
        #     outputs = net(inputs.to(device))
        #     validation_loss = criterion(outputs, labels.squeeze().long().to(device))

        valid_accuracy,validation_loss = evaluate(net, validloader, "valid", trial_num, criterion)

        # save best trained paramsclear
        print("save best net at trial_num {}".format(trial_num))
        torch.save(
                net.state_dict(),
                args.output_path + "/best_net_{:03}".format(trial_num) + ".pth.tar",
        )

        print(
            "[{:03d} / {:03d}] train_loss, valid_loss, valid_accuracy : {:.3f}, {:.3f}, {:.3f}".format(
                epoch + 1, args.epochs, train_loss, validation_loss, valid_accuracy
            )
        )

        if args.tensorboard:
            # writer.add_scalar('Loss/train', train_loss.avg, epoch)
            writer.add_scalars(
                "Loss/train", {"trial_{:03d}".format(trial_num): train_loss}, epoch
            )
            writer.add_scalars(
                "Loss/valid", {"trial_{:03d}".format(trial_num): validation_loss}, epoch
            )
            writer.add_scalars(
                "Accuracy/valid",
                {"trial_{:03d}".format(trial_num): valid_accuracy},
                epoch,
            )

            writer.flush()

    return valid_accuracy


def main():
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)

    if args.tensorboard:
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join("runs", current_time)
        writer = SummaryWriter(log_dir=log_dir)
        writer.add_text("args", str(args), 0)

    writer = writer if args.tensorboard else None

    # check train and test images
    checkimages(train_loader, writer, mode="train")
    checkimages(test_loader, writer, mode="test")

    if args.optuna:
        import logging
        import sys


        # Add stream handler of stdout to show the messages
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        study_name = "example-study"  # Unique identifier of the study.
        storage_name = "sqlite:///classifier-st.db".format(study_name)
        study = optuna.create_study(study_name=study_name, storage=storage_name,direction="maximize")
        study.optimize(
            objective_variable(train_loader, test_loader, writer),
            n_trials=args.optuna_trialnum,
        )
        print("Best params : {}".format(study.best_params))
        print("Best value  : {}".format(study.best_value))
        print("Best trial  : {}".format(study.best_trial))

        df = study.trials_dataframe()
        print(df)

        if args.tensorboard:
            df_records = df.to_dict(orient="records")
            for i in range(len(df_records)):
                df_records[i]["datetime_start"] = df_records[i][
                    "datetime_start"].strftime("%Y-%m-%d %H:%M:%S")
                df_records[i]["datetime_complete"] = df_records[i][
                    "datetime_complete"].strftime("%Y-%m-%d %H:%M:%S")
                value = df_records[i]["value"]
                value_dict = {"value": value}
                writer.add_hparams(df_records[i], value_dict)

    #     else:
    #         net = Net().to(device)
    #         criterion = nn.CrossEntropyLoss()
    #         optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    #         train(
    #             net, train_loader, test_loader, optimizer, criterion, writer, trial_num
    #         )
    # best_net = Net().to(device)
    # best_net.load_state_dict(
    #     torch.load(
    #         args.path_of_results + "/best_net_{:03d}".format(best_num) + ".pth.tar"
    #     )
    # )

    # # Read test data
    # dataiter = iter(test_loader)
    # images, labels = dataiter.next()
    # del dataiter

    # # Compare groundtruth to predicted
    # outputs = best_net(images.to(device))
    # _, predicted = torch.max(outputs, 1)

    # # Evaluate
    # test_accuracy = evaluate(best_net, test_loader, "test", best_num, criterion)
    # print("Test accuracy of best net : {:.3f}".format(test_accuracy))

    # if args.tensorboard:
    #     writer.add_scalars(
    #         "Accuracy/test/all",
    #         {"trial_{:03d}".format(best_num): test_accuracy},
    #         best_num,
    #     )

        writer.flush()
        writer.close()


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_flag = "pathmnist"
# data_flag = 'breastmnist'
download = True


lr = 0.001

info = INFO[data_flag]
task = info["task"]

n_channels = info["n_channels"]
n_classes = len(info["label"])
classes = info["label"]
DataClass = getattr(medmnist, info["python_class"])
# data_transform = transforms.Compose(
#     [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
# )
train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(
            28
        ),  # Randomly crop the image and resize it to 224x224
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(
            mean=[0.5], std=[0.5]
        ),  # Normalize the pixel values to the range [0, 1]
    ]
)

# Define data transformations for validation/testing
val_transform = transforms.Compose(
    [
        transforms.Resize(28),  # Resize the image to 256x256
        transforms.CenterCrop(28),  # Crop the center of the image to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)

# load the data
train_dataset = DataClass(split="train", transform=train_transform, download=download)
test_dataset = DataClass(split="test", transform=val_transform, download=download)
pil_dataset = DataClass(split="train", download=download, transform=None)

train_loader = data.DataLoader(
    dataset=train_dataset, batch_size=args.batch_size, shuffle=True
)

train_loader_at_eval = data.DataLoader(
    dataset=pil_dataset, batch_size=args.batch_size, shuffle=False
)
test_loader = data.DataLoader(
    dataset=test_dataset, batch_size=args.batch_size, shuffle=False
)

main()
