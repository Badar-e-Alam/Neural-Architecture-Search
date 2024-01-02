import sys
import torch
sys.path.append("..")  # Add parent folder to the sys.path

from hw_nas_bench_api import HWNASBenchAPI as HWAPI
import json
from hw_nas_bench_api.nas_201_models import get_cell_based_tiny_net
from nas_201_api import NASBench201API as API
# Create an API without the verbose log

def make_model(path="HW-NAS-Bench-v1_0.pickle",index=0,data_set="cifar100" ):
    hw_api = HWAPI(path, search_space="nasbench201")
    with open("seleted_network.json", "r") as f:
        network = json.load(f)
    config = hw_api.get_net_config(index, data_set)
    single_network = get_cell_based_tiny_net(config)#
    return single_network

# def count_layers(net):
#     return sum(1 for _ in net.modules())

def count_layers(model):
    conv_count = 0
    linear_count = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            conv_count += 1
        elif isinstance(module, torch.nn.Linear):
            linear_count += 1
    return conv_count, linear_count

def compare_model(model1,model2):
    conv_count1, linear_count1 = count_layers(model1)
    conv_count2, linear_count2 = count_layers(model2)
    if conv_count1==conv_count2 and linear_count1==linear_count2:
        return True
    else:
        return False

if __name__=="__main__":
    api = API('NAS-Bench-201-v1_0-e61699.pth', verbose=False)
# The default path for benchmark file is '{:}/{:}'.format(os.environ['TORCH_HOME'], 'NAS-Bench-201-v1_1-096897.pth')
    config  = api.get_net_config(128, 'cifar10-valid')#
    import pdb; pdb.set_trace()
    light_model=make_model(index=128,data_set="cifar10")
 # obtain the network configuration for the 123-th architecture on the CIFAR-10 dataset
    network = get_cell_based_tiny_net(config) # create the network from configurration#

    print(network) # show the structure of this architecture    # dataset=["cifar10","cifar100","ImageNet16-120"]
    status=compare_model(network,light_model)
    print(status)
    import pdb; pdb.set_trace()
        # for i in range(15625):
    #     for j in dataset:
    #         model = make_model(index=i,data_set=j)
    #         print("index:",i,"dataset:",j,"layers:",count_layers(model))
    #         import pdb; pdb.set_trace()
    #         print(model)
    # model = make_model()
    # print(model)
    # print("done")
