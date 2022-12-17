# 代码来源： https://nextjournal.com/gkoehler/pytorch-mnist 
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import time
try:
    os.mkdir("./results")
except:
    pass

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.50
momentum = 0.01
log_interval = 10

USE_CUDA = torch.cuda.is_available() and True
device = torch.device("cuda" if USE_CUDA else "cpu")
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)
train_data = torchvision.datasets.MNIST('./files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))
#train_data.data = train_data.data.to(device) # To Cuda
#train_data.targets = train_data.targets.to(device)
train_loader = torch.utils.data.DataLoader(train_data,
  batch_size=batch_size_train, shuffle=True)
test_data = torchvision.datasets.MNIST('./files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                                 # Normalize是把图像数据从[0,1]变成[-1,1]，
                                 # 两个数分别为均值(mean)和方差(std)，
                                 # 公式为：image=(image-mean)/std
                             ]))
#test_data.data = test_data.data.to(device) # To Cuda
#test_data.targets = test_data.targets.to(device)
test_loader = torch.utils.data.DataLoader(test_data,
  batch_size=batch_size_test, shuffle=True)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
#example_data = example_data.to(device)
#example_targets = example_targets.to(device)

fig = plt.figure()
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0].cpu(), cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
# fig.show()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # 输入channels=1，输出=10，卷积核的尺寸=5
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # 2D卷积操作
        self.conv2_drop = nn.Dropout2d() # dropout,随机丢弃防过拟合
        self.fc1 = nn.Linear(320, 50) # 全连接层1
        self.fc2 = nn.Linear(50, 10) # 全连接层2

    def forward(self, x):
        '''
<class 'torch.Tensor'> torch.Size([64, 1, 28, 28]) # input

<class 'torch.Tensor'> torch.Size([64, 10, 12, 12]) # 卷积1
<class 'torch.Tensor'> torch.Size([64, 20, 4, 4]) # 卷积2+丢弃

<class 'torch.Tensor'> torch.Size([64, 320]) # 压成一维

<class 'torch.Tensor'> torch.Size([64, 50]) # 全连接层1
<class 'torch.Tensor'> torch.Size([64, 50])

<class 'torch.Tensor'> torch.Size([64, 10]) # 全连接层2，输出
<class 'torch.Tensor'> torch.Size([64, 10])
        '''
        print=lambda *arg:1
        print("===")
        print(type(x),x.size())
        
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        print(type(x),x.size())
        
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        print(type(x),x.size())
        
        x = x.view(-1, 320) # 更改维度为(batch_size_train x 320)
        print(type(x),x.size())
        
        x = F.relu(self.fc1(x))
        print(type(x),x.size())
        
        x = F.dropout(x, training=self.training) # 训练的时候要用dropout，验证/测试的时候要关dropout，防过拟合
        print(type(x),x.size())
        
        x = self.fc2(x)
        print(type(x),x.size())
        
        ret = F.log_softmax(x, dim=1)
        # softmax: 归一化，转化为一个[0,1]之间的数值，这些数值可以被当做概率分布；
        # 将差距大的数值距离拉的更大（https://zhuanlan.zhihu.com/p/95415762 ）
        # log：就是自然对数
        # log_softmax(x) = log(softmax(x))
        print(type(ret),ret.size())
        
        print("===")
        #time.sleep(2)
        return ret
network = Net()
network.to(device)
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum) # 神经网络优化器，随机梯度下降，并使用动量加速
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
  # batch_id:从0开始的组数；data:图像的Tensor；target：答案的Tensor
  # (2, torch.Size([64, 1, 28, 28]), torch.Size([64]))
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad() # 梯度初始化为零
        output = network(data) # 前向传播求出预测的值
        # 这里target是一个一维数组，范围为0~9；目标是output[i][target[i]]尽可能大（也就是估计概率更高）
        # 将这些数取平均后再加负号，得到loss
        loss = F.nll_loss(output, target) # 求loss，reduction默认为'mean'
        #print(loss)
        #tensor(1.3949, grad_fn=<NllLossBackward0>)
        loss.backward() # 反向传播求梯度
        optimizer.step() # 更新所有参数
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
#            print(output,target)
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            torch.save(network.state_dict(), './results/model.pth')
            torch.save(optimizer.state_dict(), './results/optimizer.pth')

def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad(): # 不需要通过验证集来更新网络
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = network(data).to(device)
            test_loss += F.nll_loss(output, target, reduction='mean').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

train(1)

test()
#for epoch in range(1, n_epochs + 1):
#  train(epoch)
#  test()
 
 