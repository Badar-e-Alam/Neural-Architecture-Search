import tqdm
from hw_nas_bench_api import HWNASBenchAPI as HWAPI

hw_api = HWAPI("HW-NAS-Bench-v1_0.pickle", search_space="nasbench201")
data = {}
import json


# Example to get all the hardware metrics in the No.0,1,2 architectures under NAS-Bench-201's Space
def read_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def write_json(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


ori_nas_bench = read_json("data.json")
config = read_json("config.json")

print(
    "===> Example to get all the hardware metrics in the No.0,1,2 architectures under NAS-Bench-201's Space"
)
latency_list = []
low_latency = 2.0
val_acc = []
dataset_list = ["cifar10", "cifar100", "ImageNet16-120"]
metrics_dict = {dataset_name: {} for dataset_name in dataset_list}

for idx in tqdm.tqdm(range(15625)):
    for dataset in config["dataset"]:
        loss_acc = ori_nas_bench[dataset][str(idx)]
        train_loss, validation_loss, train_accuracy, validation_accuracy = loss_acc[:]
        HW_metrics = hw_api.query_by_index(idx, dataset)
        latency_list.append(HW_metrics["fpga_latency"])
        val_acc.append(validation_accuracy)
        if (
            HW_metrics["fpga_latency"] > config["fpga_latency"]
            and validation_accuracy > config["validation_accuracy"]
        ):
            metrics_dict[dataset][idx] = {
                "fpga_latency": HW_metrics["fpga_latency"],
                "accuracy": validation_accuracy,
            }
            import pdb ; pdb.set_trace()
            # print("HW_metrics (type: {}) for No.{} @ {} under NAS-Bench-201: {}".format(type(HW_metrics), idx, dataset, HW_metrics["fpga_energy"]))

print(
    f"median latency {sorted(latency_list)[len(latency_list)//2]} median acc {sorted(val_acc)[len(val_acc)//2]}"
)
# print(f"mean latency {sum(latency_list)/len(latency_list)} max latency {max(latency_list)} min latency {min(latency_list)} mean val_acc {sum(val_acc)/len(val_acc)} max val_acc {max(val_acc)} min val_acc {min(val_acc)}")

write_json("metrics.json", metrics_dict)
# print("The HW_metrics (type: {}) for No.{} @ {} under NAS-Bench-201: {}".format(type(HW_metrics),
#                                                                        idx,ls

#                                                                        dataset,
#                                                                        HW_metrics["fpga_energy"]))H!_43ugW
