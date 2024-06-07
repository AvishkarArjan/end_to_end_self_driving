#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

class DAVE2(nn.Module):
	def __init__(self):
		super(DAVE2,self).__init__()
		self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
		self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
		self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
		self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=2)
		self.flatten = nn.Flatten()
		self.dropout1 = nn.Dropout(0.3)
		self.fc1 = nn.Linear(64 * 2* 2, 100)
		self.dropout2 = nn.Dropout(0.3)
		self.fc2 = nn.Linear(100, 50)
		self.dropout3 = nn.Dropout(0.2)
		self.fc3 = nn.Linear(50, 10)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		# print(x.shape)
		x = self.flatten(x)
		x = self.dropout1(x)
		x = F.relu(self.fc1(x))
		x = self.dropout2(x)
		x = F.relu(self.fc2(x))
		x = self.dropout3(x)
		x = self.fc3(x)
		return x

class DAVE2_D(nn.Module):
	def __init__(self):
		super(DAVE2_D,self).__init__()
		self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
		self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
		self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
		self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=2)
		self.flatten = nn.Flatten()
		self.dropout1 = nn.Dropout(0.3)
		self.fc1 = nn.Linear(64 * 2* 10, 100)
		# self.fc1 = nn.Linear(64 * 6* 6, 100)
		self.dropout2 = nn.Dropout(0.3)
		self.fc2 = nn.Linear(100, 50)
		self.dropout3 = nn.Dropout(0.2)
		self.fc3 = nn.Linear(50, 1)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		# print(x.shape)
		x = self.flatten(x)
		x = self.dropout1(x)
		x = F.relu(self.fc1(x))
		x = self.dropout2(x)
		x = F.relu(self.fc2(x))
		x = self.dropout3(x)
		x = self.fc3(x)
		return x

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape)
        # x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = nn.Flatten()(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Net_D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 64, 5)
        self.fc1 = nn.Linear(64 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape)
        # x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = nn.Flatten()(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
	model = DAVE2_D()
	# input_ = torch.randn(32,3,128,128) # (Bs, c, h, w)
	input_ = torch.randn(32,3,66,200) # (Bs, c, h, w)
	out = model(input_)
	print(out.shape)