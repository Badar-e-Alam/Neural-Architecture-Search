import torch.nn as nn

class Image_classification(nn.Module):
    def __init__(self):
        super(Image_classification, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=5)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(576, 256)  # Adjust the in_features based on the output size after convolution
        self.relu4 = nn.ReLU()

        self.dropout = nn.Dropout(0.5)  # Add dropout layer for regularization

        self.fc2 = nn.Linear(256, 128)
        self.relu5 = nn.ReLU()

        self.fc3 = nn.Linear(128, 4)  # Adjust out_features to be 4

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)

        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu5(x)

        x = self.fc3(x)
        x = self.softmax(x)

        return x

# Create an instance of the modified complex model
