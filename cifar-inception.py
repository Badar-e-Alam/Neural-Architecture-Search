import random
import tqdm
from deap import base, creator, tools, algorithms
import torch
import torch.nn as nn
import os
import time
import json
import multiprocessing as mp
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.nn import functional as F
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
from torchvision import transforms
from pytorchcv.model_provider import _models
from seperate import insert_sparsity, calculate_fitness
from torch.utils.tensorboard import SummaryWriter
import ffcv
import ffcv.fields as fields
import ffcv.fields.decoders as decoders
import ffcv.transforms as transforms

writer = SummaryWriter("./Charts")

# Load the pre-trained Inception model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_saved = False
resnet_model = _models["resnet20_cifar10"](pretrained=True)
# import pdb;pdb.set_trace()
# inception_v3 = torchvision.models.inception_v3(pretrained=True)
# for params in inception_v3.parameters():
#     params.requires_grad=False
# inception_v3.fc=nn.Linear(in_features=2048,out_features=10) #as per cifar10
# Define the data transforms
"""transform = transforms.Compose(
    # [   transforms.Resize(299),
    #     transforms.CenterCrop(299),
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)"""
print(f"Code is running on {device}")
# Load the CIFAR10 dataset
"""train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=200, shuffle=True)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)
"""
DATA_DIR = "./data"

NUM_CLASSES = 10
CIFAR_MEAN = [0.485, 0.456, 0.406]
CIFAR_STD = [0.229, 0.224, 0.225]

BATCH_SIZE = 512
NUM_WORKERS = 8
# Loss function and optimizer
criterion = nn.CrossEntropyLoss()


def convert_dataset(dset, name):
    writer = ffcv.writer.DatasetWriter(
        name + ".beton", {"image": fields.RGBImageField(), "label": fields.IntField()}
    )
    writer.from_indexed_dataset(dset)


train_dset = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=True)
test_dset = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True)
convert_dataset(train_dset, "cifar_train")
convert_dataset(test_dset, "cifar_test")


def get_image_pipeline(train=True):
    augmentation_pipeline = (
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomTranslate(padding=2),
            transforms.Cutout(8, tuple(map(int, CIFAR_MEAN))),
        ]
        if train
        else []
    )

    image_pipeline = (
        [decoders.SimpleRGBImageDecoder()]
        + augmentation_pipeline
        + [
            transforms.ToTensor(),
            transforms.ToDevice(device, non_blocking=True),
            transforms.ToTorchImage(),
            transforms.Convert(torch.float32),
            torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    return image_pipeline


label_pipeline = [
    decoders.IntDecoder(),
    transforms.ToTensor(),
    transforms.ToDevice(device),
    transforms.Squeeze(),
]


train_image_pipeline = get_image_pipeline(train=True)
test_image_pipeline = get_image_pipeline(train=False)
train_loader = ffcv.loader.Loader(
    f"cifar_train.beton",
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    order=ffcv.loader.OrderOption.RANDOM,
    drop_last=True,
    pipelines={"image": train_image_pipeline, "label": label_pipeline},
)


test_loader = ffcv.loader.Loader(
    f"cifar_test.beton",
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    order=ffcv.loader.OrderOption.SEQUENTIAL,
    drop_last=False,
    pipelines={"image": test_image_pipeline, "label": label_pipeline},
)

iteration = 0


def append_to_json_file(
    param1, param2, param3, param4, param5, file_path="logging_4_ffcv.json"
):
    param1 = (
        convert_tensor_to_list(param1) if isinstance(param1, torch.Tensor) else param1
    )
    param3 = (
        convert_tensor_to_list(param3) if isinstance(param3, torch.Tensor) else param3
    )
    param4 = (
        convert_tensor_to_list(param4) if isinstance(param4, torch.Tensor) else param4
    )
    try:
        # Load existing data from the file, if any
        with open(file_path, "r") as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        # If the file doesn't exist, initialize with an empty list
        existing_data = []

    # Append new data to the existing data
    timestamp = datetime.now().isoformat()
    existing_data.append(
        {
            "Train Loss": param1,
            "Speed": param2,
            "Validation Accuracy": param3,
            "Validation Loss": param4,
            "Timestamps": timestamp,
            "Sparcity_list": param5,
        }
    )

    # Write the updated data back to the file
    with open(file_path, "w") as file:
        json.dump(existing_data, file, indent=2)


def convert_tensor_to_list(tensor):
    """Convert a PyTorch Tensor to a nested list."""
    if tensor.dim() == 0:
        return tensor.item()
    return [convert_tensor_to_list(x) for x in tensor]


# running_loss=0.0
# iteration=0
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm.tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return (running_loss / len(train_loader)), model


# Validation loop
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in tqdm.tqdm(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    val_loss = running_loss / len(val_loader)
    val_accuracy = correct_preds / total_samples

    return val_loss, val_accuracy


iteration = 0.0


def normalize(x):
    min_value = min(x)
    max_value = max(x)
    normalized = [(float(i) - min_value) / (max_value - min_value) for i in x]
    return normalized


def evaluateModel(sparcity_list):
    # optimization function
    global iteration
    iteration += 1
    criterion = nn.CrossEntropyLoss()
    pruned_model = insert_sparsity(resnet_model, sparsity_list=normalize(sparcity_list))
    optimizer = torch.optim.Adam(pruned_model.parameters(), lr=1e-2)
    fitness_scr = calculate_fitness(
        pruned_model.to(device), sparsity_list=normalize(sparcity_list)
    )
    num_epochs = 1

    for epoch in range(num_epochs):
        """
        code is write stopping here but if you check inside the traina nd validation
        both will tell you data and model both are on the cuda. but for myside problem is that script is running really slow
        as its running on CPU also the GPU consummption is around 1-2 % when code is running.
        """

        train_loss, t_model = train(
            pruned_model, train_loader, criterion, optimizer, device
        )
        val_loss, val_accuracy = validate(t_model, test_loader, criterion, device)

        print(
            f"Epoch {epoch + 1}/{num_epochs} => "
            f"Train Loss: {train_loss:.4f}, "
            f"Validation Loss: {val_loss:.4f}, "
            f"Validation Accuracy: {val_accuracy:.4f}"
        )

        print("Iteration", iteration)

        writer.add_scalar("Training loss loss ", train_loss, iteration)
        writer.add_scalar("Validation loss ", val_loss, iteration)
        writer.add_scalar("Validation accuracy  ", val_accuracy, iteration)
        writer.add_scalar("Fitness Score ", fitness_scr, iteration)
        writer.flush()

        append_to_json_file(
            train_loss, fitness_scr, val_accuracy, val_loss, normalize(sparcity_list)
        )

    return (fitness_scr, train_loss, val_loss, val_accuracy)


def count_conv2d_and_linear_layers(model):
    """Counts the number of convolutional and linear layers in a PyTorch model.

    Args:
      model: A PyTorch model.

    Returns:
      A tuple containing the number of convolutional and linear layers in the model.
    """
    num_conv2d_layers = 0
    num_linear_layers = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            num_conv2d_layers += 1
        elif isinstance(module, torch.nn.Linear):
            num_linear_layers += 1
    return num_conv2d_layers + num_linear_layers


population_size = 100

if __name__ == "__main__":
    No_cromo = count_conv2d_and_linear_layers(resnet_model)
    Num_process = 8
    print(
        f"No of cromosomes {No_cromo}"
    )  # (fitness_scr,train_loss, val_loss, val_accuracy)
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, -1.0, 1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 1, 5)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_bool,
        n=No_cromo,
    )  # one genes value
    # toolbox.register("attr_bool", random.uniform, 0, 1)
    # toolbox.register(
    #     "individual",
    #     tools.initRepeat,
    #     creator.Individual,
    #     toolbox.attr_bool,
    #     n=No_cromo,
    # )  # one genes value

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluateModel)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.09)
    toolbox.register("select", tools.selTournament, tournsize=4)
    pool = mp.Pool(processes=Num_process)

    population = toolbox.population(n=population_size)  # total genes  list of list

    NGEN = 10
    for gen in range(NGEN):
        starttime = time.time()
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
        # offspring = [[float(gene) for gene in ind] for ind in offspring]
        # offspring_tensors = [torch.tensor(individual) for individual in offspring]

        # # Move offspring tensors to the same device as the model
        # offspring_tensors = [tensor.to(device) for tensor in offspring_tensors]

        fits = pool(list(map(toolbox.evaluate, offspring)))

        endtime = time.time() - starttime
        print(f"It took {endtime} to run the process")
        # here have the multiple options to achieve our desire result, save the fits score along with the offspring genes.
        for fit, ind in tqdm.tqdm(zip(fits, offspring)):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))

    top10 = tools.selBest(population, k=10)
    with open("result.json", "w") as jsfile:
        json.dump(top10, jsfile, indent=2)

    print(top10)
