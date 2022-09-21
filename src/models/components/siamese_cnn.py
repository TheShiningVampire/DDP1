from torch import nn
import torchvision

class Siamese_CNN(nn.Module):
    def __init__(
        self,
        input_channels: int,
        input_size: int,
        ):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, input_channels*2, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d( input_channels*2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(input_channels*2, input_channels*4, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(input_channels*4)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Use the FC layers of a pretrained network (ResNet-18)
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.avgpool = resnet50.avgpool
        self.fc1 = resnet50.fc
        self.dropout1 = nn.Dropout(p=0.5)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(1000, 256)


    def forward_once(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.avgpool(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.relu3(x)
        x = self.fc2(x)

        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2


if __name__ == "__main__":
    _ = Siamese_CNN(3, 32)
