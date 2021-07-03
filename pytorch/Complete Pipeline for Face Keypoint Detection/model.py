import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.nn.init as I 

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1,32,5)
        self.pool1 = nn.MaxPool2d(2,2)
        self.drop1 = nn.Dropout(p=0.1)

        self.conv2 = nn.Conv2d(32,64,3)
        self.pool2 = nn.MaxPool2d(2,2)
        self.drop2 = nn.Dropout(p=0.2)

        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool3 = nn.MaxPool2d(2,2)
        self.drop3 = nn.Dropout(p=0.3)

        self.conv4 = nn.Conv2d(128, 256, 3)
        self.pool4 = nn.MaxPool2d(2,2)
        self.drop4 = nn.Dropout(p=0.4)

        self.conv5 = nn.Conv2d(256, 512, 3)
        self.pool5 = nn.MaxPool2d(2,2)
        self.drop5 = nn.Dropout(p=0.5)

        self.fc6 = nn.Linear(512*5*5, 2560)
        self.drop6 = nn.Dropout(p=0.4)
        
        self.fc7 = nn.Linear(2560, 1280)
        self.drop7 = nn.Dropout(p=0.4)
        
        self.fc8 = nn.Linear(1280, 136)
    
    def forward(self,x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.drop1(x)

        x = self.pool2(F.relu(self.conv2(x)))
        x = self.drop2(x)

        x = self.pool3(F.relu(self.conv3(x)))
        x = self.drop3(x)
        
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.drop4(x)
        
        x = self.pool5(F.relu(self.conv5(x)))
        x = self.drop5(x)
        
        #print("size: ",x.size())
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc6(x))
        x = self.drop6(x)
        
        x = F.relu(self.fc7(x))
        x = self.drop7(x)
        
        x = self.fc8(x)

        return x
