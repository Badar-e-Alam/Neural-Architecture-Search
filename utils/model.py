from hw_nas_bench_api import HWNASBenchAPI as HWAPI
import torch
from torch import nn
import json
from hw_nas_bench_api.nas_201_models import get_cell_based_tiny_net

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


class Net(nn.Module):
    def __init__(self, dropout_enabled=True):
        super(Net, self).__init__()
        self.model = single_network
        self.dropout_enabled = dropout_enabled
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        x = self.model(x)
        if self.dropout_enabled:
            x = self.dropout(x[1])
        return x
