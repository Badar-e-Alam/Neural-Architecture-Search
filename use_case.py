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
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
config = hw_api.get_net_config(selected_network[-2], "cifar10")
model = get_cell_based_tiny_net(config)

##loading dataset

data_flag = 'pathmnist'
# data_flag = 'breastmnist'
download = True

NUM_EPOCHS = 3
BATCH_SIZE = 128
lr = 0.001

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# load the data
train_dataset = DataClass(split='train', transform=data_transform, download=download)
test_dataset = DataClass(split='test', transform=data_transform, download=download)

pil_dataset = DataClass(split='train', download=download)

train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)



print(train_dataset)
print("===================")
print(test_dataset)

#defining the model

class Net(nn.Module):
        def __init__(self):
                super(Net, self).__init__()
                self.model = model
        
        def forward(self, x):
                x = self.model(x)
                return x
# define loss function and optimizer
if task == "multi-label, binary-class":
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()
    
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

for epoch in range(NUM_EPOCHS):
    train_correct = 0
    train_total = 0
    test_correct = 0
    test_total = 0
    
    model.train()
    for inputs, targets in tqdm.tqdm(train_loader):
        # forward + backward + optimize
        optimizer.zero_grad()
        outputs = model(inputs)
        
        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32)
            loss = criterion(outputs, targets)
        else:
            targets = targets.squeeze().long()
            loss = criterion(outputs[1], targets)
        
        loss.backward()
        optimizer.step()



# evaluation

def test(split):
    model.eval()
    y_true = torch.tensor([])
    y_score = torch.tensor([])
    
    data_loader = train_loader_at_eval if split == 'train' else test_loader

    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                outputs = outputs.softmax(dim=-1)
            else:
                targets = targets.squeeze().long()
                outputs = outputs.softmax(dim=-1)
                targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.numpy()
        y_score = y_score.detach().numpy()
        
        evaluator = Evaluator(data_flag, split)
        metrics = evaluator.evaluate(y_score)
    
        print('%s  auc: %.3f  acc:%.3f' % (split, *metrics))

        
print('==> Evaluating ...')
test('train')
test('test')

