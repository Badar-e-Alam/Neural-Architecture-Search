from pytorchcv.model_provider import get_model as ptcv_get_model
import torch
from torchviz import make_dot
from torch.autograd.profiler import profile
from torch import nn
from torch.nn import functional as F
import tqdm
import hiddenlayer as hl
from torchsummary import summary
import multiprocessing
import netron
from torch.utils.tensorboard import SummaryWriter
from graphviz import Digraph


class resnet18(nn.Module):
    def __init__(self):
        super(resnet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1000)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, inplanes, planes, blocks, stride=1):
        layers = []
        layers.append(
            nn.Conv2d(
                inplanes,
                planes,
                kernel_size=(3, 3),
                stride=(stride, stride),
                padding=(1, 1),
            )
        )
        layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.ReLU(inplace=True))
        for i in range(1, blocks):
            layers.append(
                nn.Conv2d(
                    planes, planes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
                )
            )
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        import pdb

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 56
        x = self.maxpool(x)  # 28
        x = self.layer1(x)  # 28
        x = self.layer2(x)  # 14
        x = self.layer3(x)  # 7
        x = self.layer4(x)  # 4
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def number_of_kernel(model):
    num = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            num += 1
    return num


def change_model_kernel(model, kernel_size=3):
    for name, m in tqdm.tqdm(model.named_modules()):
        if isinstance(m, nn.Conv2d):
            # Modify the kernel size of the convolutional layer
            m.kernel_size = (kernel_size, kernel_size)
            m.padding = (kernel_size // 2, kernel_size // 2)
            conv_output_size = (
                m.in_channels - kernel_size + 2 * m.padding[0]
            ) // m.stride[0] + 1
            m.out_channels = (
                m.out_channels * conv_output_size * conv_output_size // m.in_channels
            )
        elif isinstance(m, nn.Linear):
            # Recalculate input size for linear layers
            m.in_features = (
                conv_output_size * conv_output_size * m.in_features // m.in_features
            )
    return model


def plot_model(model, name, input_shape):
    dot = Digraph(comment="Model Architecture")
    dot.attr("node", shape="box")
    dot.node("input", f"Input\n{input_shape}")
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            dot.node(
                name,
                f"Conv2d\n{module.kernel_size[0]}x{module.kernel_size[1]}\n{module.out_channels} filters",
            )
        elif isinstance(module, nn.Linear):
            dot.node(name, f"Linear\n{module.in_features}x{module.out_features}")
        elif isinstance(module, nn.BatchNorm2d):
            dot.node(name, "BatchNorm2d")
        elif isinstance(module, nn.ReLU):
            dot.node(name, "ReLU")
        else:
            dot.node(name, str(module))

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            for name2, module2 in model.named_modules():
                if module2 == module:
                    dot.edge(name2, name)
        elif isinstance(module, nn.Linear):
            for name2, module2 in model.named_modules():
                if module2 == module:
                    dot.edge(name2, name)

    dot.edge("input", "conv1")
    dot.format = "pdf"

    dot.render(name, view=True)


def start_netron(model_path):
    netron.start(model_path)


if __name__ == "__main__":
    # Modify the model kernel size
    writer1 = SummaryWriter("runs/exp1")
    writer2 = SummaryWriter("runs/exp2")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ptcv_get_model("resnet18", pretrained=True)
    model1 = resnet18().to(device)
    num_kernels = number_of_kernel(model)
    print("Number of kernels: ", num_kernels)

    update_model = change_model_kernel(model, kernel_size=7)
    import pdb

    pdb.set_trace()

    input = torch.randn(1, 3, 224, 224).to(device)
    output = model1(input)
    output2 = update_model(input)
    writer1.add_graph(model, input)
    writer2.add_graph(update_model, input)

    writer1.close()
    writer2.close()
    # plot_model(model1,"basic_model" ,input.shape)
    # plot_model(update_model,"updated_model", input.shape)
    # import pdb; pdb.set_trace()
    # graph = hl.build_graph(model, torch.zeros([1, 3, 224, 224]))
    # graph.theme = hl.graph.THEMES["blue"].copy()
    # graph.save("model_graph", format="png")

    graph1 = make_dot(output, params=dict(model.named_parameters()))
    graph1.format = "pdf"
    graph1.render("ResNet18", cleanup=True)
    graph2 = make_dot(output2, params=dict(model.named_parameters()))
    graph2.format = "pdf"
    graph2.render("ResNet18_updated", cleanup=True)
    summary(model1, (3, 224, 224))
    summary(update_model, (3, 224, 224))
    traced_script_module1 = torch.jit.trace(model1, input)
    traced_script_module1.save("traced_model1.pt")

    traced_script_module2 = torch.jit.trace(update_model, input)
    traced_script_module2.save("traced_model2.pt")

    p1 = multiprocessing.Process(target=start_netron, args=("traced_model1.pt",))
    p2 = multiprocessing.Process(target=start_netron, args=("traced_model2.pt",))

    p1.start()
    p2.start()

    # Recalculate the number of output and input sizes for the model's other layers
