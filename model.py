import torch
from torchvision import datasets, transforms
from torch.utils.data import dataloader
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import time
import torch.nn.init as init

#preprocess the images
train_transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(size=32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#make dataloader for training and testing
train_data = datasets.CIFAR10(root='./data', train= True, transform= train_transform, download= True)
test_data = datasets.CIFAR10(root='./data', train= False, transform= test_transform, download= True)
train_data_size = len(train_data)
test_data_size = len(test_data)
train_loader = dataloader.DataLoader(dataset= train_data, batch_size=128, shuffle=True, num_workers= 2)
test_loader = dataloader.DataLoader(dataset= test_data, batch_size=128, num_workers= 2)

use_gpu = torch.cuda.is_available()

class ResBlock(nn.Module):
    """The block of residual network"""

    def __init__(self, in_channel, out_channel, size):
        super(ResBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.size = size
        self.conv1 = torch.nn.Conv2d(in_channel, out_channel, (3, 3), padding= 1)
        self.conv1_ = torch.nn.Conv2d(in_channel, out_channel, (3, 3), stride= 2, padding= 1)
        self.conv2 = torch.nn.Conv2d(out_channel, out_channel, (3, 3), padding= 1)
        self.bn = torch.nn.BatchNorm2d(out_channel)
        self.relu = torch.nn.PReLU()
        self.max_pool = torch.nn.MaxPool2d(2, 2)

    def forward(self, input):
        identity = input
        #if the channels and image size would not change after the block
        if self.in_channel == self.out_channel:
            output = self.conv1(input)
            output = self.bn(output)
            output = self.relu(output)
            output = self.conv2(output)
            output = self.bn(output)
            output = torch.add(output, identity)
            output = self.relu(output)
        else:
            #if the channels and image size would change after the block
            identity = self.max_pool(identity)
            identity = torch.cat((identity, Variable(torch.zeros(identity.size()))), 1)
            output = self.conv1_(input)
            output = self.bn(output)
            output = self.relu(output)
            output = self.conv2(output)
            output = self.bn(output)
            output = torch.add(output, identity)
            output = self.relu(output)

        return output

class ResNet(nn.Module):
    """The architecture of residual network"""

    def __init__(self):
        super(ResNet, self).__init__()
        self.relu = torch.nn.PReLU()
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.conv1 = torch.nn.Conv2d(3, 16, (3, 3), padding= 1)
        self.res_block1 = ResBlock(16, 16, 32)
        self.res_block2 = ResBlock(16, 16, 32)
        self.res_block3 = ResBlock(16, 16, 32)
        self.res_block4 = ResBlock(16, 32, 32)
        self.res_block5 = ResBlock(32, 32, 16)
        self.res_block6 = ResBlock(32, 32, 16)
        self.res_block7 = ResBlock(32, 64, 16)
        self.res_block8 = ResBlock(64, 64, 8)
        self.res_block9 = ResBlock(64, 64, 8)
        self.global_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(64, 10)
        self.softmax = torch.nn.Softmax()
        self.init_params()

    def init_params(self):
        """Initialize the parameters in the ResNet model"""

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            if isinstance(m, nn.Linear):
                init.normal(m.weight, std= 0.01)
                if m.bias is not None:
                    init.constant(m.bias, 0)

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.res_block1(output)
        output = self.res_block2(output)
        output = self.res_block3(output)
        output = self.res_block4(output)
        output = self.res_block5(output)
        output = self.res_block6(output)
        output = self.res_block7(output)
        output = self.res_block8(output)
        output = self.res_block9(output)
        output = self.global_pool(output)
        output = output.view(-1, 64)
        output = self.fc(output)
        output = self.softmax(output)

        return output

def lr_optimizer(optimizer, step):
    """The schedule of changing the learning rate

       When the step is 32000 or 48000, the learning rate would be divided by 10

    :param: optimizer: the optimizer to update the parameters
    :param: step: the training iterations
    """

    if step == 32000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 10
        print('Learning rate is set to {: 4f}'.format(optimizer.param_groups[0]['lr']))
    elif step == 48000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 10
        print('Learning rate is set to {: 4f}'.format(optimizer.param_groups[0]['lr']))
    return optimizer

def train(res_net, lr_optimizer, optimizer, criterion, train_loader, max_epoch = 160):
    """Train the ResNet model"""

    step = 0
    start_time = time.time()
    for epoch in range(max_epoch):
        running_loss = 0.0
        for data in train_loader:
            step += 1
            train_input, train_label = data
            if use_gpu:
                train_input, train_label = Variable(train_input.cuda()), Variable(train_label.cuda())
            else:
                train_input, train_label = Variable(train_input), Variable(train_label)

            optimizer = lr_optimizer(optimizer, step)
            optimizer.zero_grad()
            train_outputs = res_net(train_input)
            loss = criterion(train_outputs, train_label)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]

            if step % 10 == 0:
                end_time = time.time()
                duration = end_time - start_time
                start_time = time.time()
                print('Step: {}, Loss: {: 4f}, Durantion per 10 steps: {: 2f}'.format(step, loss.data[0], duration))

        epoch_loss = running_loss / (train_data_size / 128)
        print('Epoch: {}, Loss: {: 4f}'.format(epoch, epoch_loss))

    return res_net

res_net = ResNet()
if use_gpu:
    res_net = res_net.cuda()
criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.SGD(res_net.parameters(), lr = 0.1, momentum= 0.9, weight_decay= 0.0001)

#the parameters of PRelu could not have weight decay
optimizer = optim.SGD([
    {'params': res_net.bn1.parameters(), 'weight_decay': 0.0001},
    {'params': res_net.conv1.parameters(), 'weight_decay': 0.0001},
    {'params': res_net.res_block1.parameters(), 'weight_decay': 0.0001},
    {'params': res_net.res_block2.parameters(), 'weight_decay': 0.0001},
    {'params': res_net.res_block3.parameters(), 'weight_decay': 0.0001},
    {'params': res_net.res_block4.parameters(), 'weight_decay': 0.0001},
    {'params': res_net.res_block5.parameters(), 'weight_decay': 0.0001},
    {'params': res_net.res_block6.parameters(), 'weight_decay': 0.0001},
    {'params': res_net.res_block7.parameters(), 'weight_decay': 0.0001},
    {'params': res_net.res_block8.parameters(), 'weight_decay': 0.0001},
    {'params': res_net.res_block9.parameters(), 'weight_decay': 0.0001},
    {'params': res_net.relu.parameters()},
], lr= 0.1, momentum= 0.9)
res_net = train(res_net, lr_optimizer, optimizer, criterion, train_loader)

#calculate the accuracy of the trained model
corrects = 0
for data in test_loader:
    test_input, test_label = data
    if use_gpu:
        test_input, test_label = Variable(test_input.cuda()), Variable(test_label.cuda())
    else:
        test_input, test_label = Variable(test_input), Variable(test_label)

    test_outputs = res_net(test_input)
    _, preds = torch.max(test_outputs.data, 1)
    corrects += torch.sum(preds == test_label.data)

accuracy = corrects / test_data_size
print('The accuracy in the test dataset is {: 4f}'.format(accuracy))

