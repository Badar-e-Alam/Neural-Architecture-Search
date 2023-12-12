from hw_nas_bench_api import HWNASBenchAPI as HWAPI
import tqdm
import matplotlib.pyplot as plt
import random
from deap import base, creator, tools, algorithms
import torch
import torch.nn as nn
import time
import json
import multiprocessing as mp
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.nn import functional as F
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from hw_nas_bench_api.nas_201_models import get_cell_based_tiny_net
from deap import base, creator, tools, algorithms
import random
import multiprocessing as mp
import time
from datetime import datetime
import json
import ffcv
from torchlop import profile

from torch.utils.tensorboard import SummaryWriter
import ffcv.fields as fields
import ffcv.fields.decoders as decoders
import ffcv.transforms as transforms
writer = SummaryWriter("runs/HWNASBench")
hw_api = HWAPI("HW-NAS-Bench-v1_0.pickle", search_space="nasbench201")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data={}
DATA_DIR = "./data"

NUM_CLASSES = 10
CIFAR_MEAN = [0.485, 0.456, 0.406]
CIFAR_STD = [0.229, 0.224, 0.225]
iteration=0
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

def plot_histogram(fpga_energy_values,label,name):
    plt.hist(fpga_energy_values, bins=20)
    plt.xlabel(label)
    plt.ylabel("Frequency")
    plt.title("Histogram of FPGA"+label)
    plt.savefig(name)

def calculate_fitness(model, sparsity_list=[0.10] * 2):
    idx_l = 0
    input_ = torch.randn(1, 3, 32, 32).to(device=device)
    macs, params, layer_infos = profile(model, inputs=(input_,))
    total_speedup = 0
    for k, v in layer_infos.items():
        if v["type"] == "Conv2d" or v["type"] == "Linear":
            sparsity = sparsity_list[idx_l]
            if v["ops"] is None:
                # print("Senario # 1 ")
                mac_weight = 0.0
            elif v["params"] is None:
                # print("Senario # 2 ")
                v["params"] = 0.0
                mac_weight = v["ops"] / 0.0

            else:
                mac_weight = v["ops"] / v["params"]
            # print(v['ops'], v['params'])
            if v["params"] is None:
                speedup = sparsity * 0.0 * mac_weight

            else:
                speedup = sparsity * v["params"] * mac_weight
            total_speedup += speedup
            idx_l += 1
    # calculate validation accuracy
    return (total_speedup) / macs

def change_model_kernel(model, sparcity_list):
    count=0
    for name, m in tqdm.tqdm(model.named_modules()):
        if isinstance(m, nn.Conv2d):
            # Modify the kernel size of the convolutional layer
            m.kernel_size = (sparcity_list[count], sparcity_list[count])
            m.padding = (sparcity_list[count] // 2, sparcity_list[count] // 2)
            conv_output_size = (
                m.in_channels - sparcity_list[count] + 2 * m.padding[0]
            ) // m.stride[0] + 1
            m.out_channels = m.out_channels * conv_output_size * conv_output_size // m.in_channels
        elif isinstance(m, nn.Linear):
            # Recalculate input size for linear layers
            m.in_features = (
                conv_output_size * conv_output_size * m.in_features // m.in_features
            )
            count+=1
        else:
            pass
    return model

def scatter_plot(fpga_energy_values, fpga_latency_values):
    plt.plot(fpga_energy_values, fpga_latency_values)
    plt.xlabel("FPGA Energy")
    plt.ylabel("FPGA Latency")
    plt.title("LIne Plot of FPGA Energy and FPGA Latency")
    plt.savefig("Line.png")                                                    
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
                        

def train(model, train_loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    for inputs, labels in tqdm.tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs[0], labels)
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
            loss = criterion(outputs[0], labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs[0], 1)
            correct_preds += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    
    val_loss = running_loss / len(val_loader)
    val_accuracy = correct_preds / total_samples

    return val_loss, val_accuracy

def convert_tensor_to_list(tensor):
    """Convert a PyTorch Tensor to a nested list."""
    if tensor.dim() == 0:
        return tensor.item()
    return [convert_tensor_to_list(x) for x in tensor]

def normalize(x):
    min_value=min(x)
    max_value=max(x)
    normalized = [(float(i)-min_value)/(max_value-min_value) for i in x]
    return normalized
def append_to_json_file(
    param1, param2, param3, param4, param5, file_path="logging.json"
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

def evaluateModel(sparcity_list):


    # optimization function
    global iteration
    iteration += 1
    customize_model = modify_model_kernel(network, sparcity_list)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(network.parameters(), lr=1e-2)
    num_epochs = 1
    customize_model.to(device)
    fitness_scr = calculate_fitness(customize_model, sparcity_list)
    for epoch in range(num_epochs):
        """
        code is write stopping here but if you check inside the traina nd validation
        both will tell you data and model both are on the cuda. but for myside problem is that script is running really slow
        as its running on CPU also the GPU consummption is around 1-2 % when code is running.
        """

        train_loss, t_model = train(
            customize_model, train_loader, criterion, optimizer, device
        )
        val_loss, val_accuracy = validate(customize_model, test_loader, criterion, device)

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

    return ( fitness_scr,train_loss, val_loss, val_accuracy)

population_size =100
Num_process = 8
def modify_model_kernel(model, kernel_sizes):
    # Function to modify kernel size for all convolutional layers
    kernel_sizes_iter = iter(kernel_sizes)
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            try:
                module.kernel_size = next(kernel_sizes_iter)
            except StopIteration:
                break
    return model

def calculate_output_size(input_size, kernel_size, stride, padding):
    # Function to calculate the output size of a convolutional layer
    return ((input_size - kernel_size + 2 * padding) // stride) + 1
       
if __name__ == "__main__":
    #Example to get all the hardware metrics in the No.0,1,2 architectures under NAS-Bench-201's Space
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, -1.0, 1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 1, 5)

    toolbox.register("evaluate", evaluateModel)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.09)
    toolbox.register("select", tools.selTournament, tournsize=4)
    with open("seleted_network.json", "r") as jsfile:
        selected_network = json.load(jsfile)
    
    import pdb ; pdb.set_trace()

    for idx in tqdm.tqdm(selected_network):
        print(f"Network index: {idx}")
        for dataset in ["cifar10"]:
            # HW_metrics = hw_api.query_by_index(idx, dataset)
            # # engery_list.append(HW_metrics["fpga_energy"])
            # # latency_list.append(HW_metrics["fpga_latency"])
            # if HW_metrics["fpga_energy"]>7.6674888908800005:
            config = hw_api.get_net_config(idx, dataset)
            network = get_cell_based_tiny_net(config) # create the network from configurration
                # data["engery"]=engery_list
                # data["latency"]=latency_list
                # plot_histogram(data["engery"],label="fpga_energy",name="energy.png")
                # plot_histogram(data["latency"],label="fpga_latency",name="latency.png")
                # scatter_plot(engery_list,latency_list)
            iteration+10
            No_cromo=count_conv2d_and_linear_layers(network)
            print(
                    f"No of cromosomes {No_cromo}"
            )  # (fitness_scr,train_loss, val_loss, val_accuracy)
    	        # total genes  list of list
            toolbox.register(
                    "individual",
                    tools.initRepeat,
                    creator.Individual,
                    toolbox.attr_bool,
                    n=No_cromo,
                )
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            population = toolbox.population(n=population_size)



            NGEN = 10
            for gen in range(NGEN):
                starttime = time.time()
                offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
                #offspring = [[float(gene) for gene in ind] for ind in offspring]
                # offspring_tensors = [torch.tensor(individual) for individual in offspring]

                # # Move offspring tensors to the same device as the model
                # offspring_tensors = [tensor.to(device) for tensor in offspring_tensors]
                            
                fits = toolbox.map(toolbox.evaluate, offspring)

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
