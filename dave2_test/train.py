#!/usr/bin/env python3

import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm
from model import DAVE2, DAVE2_D, Net, Net_D
from data import get_cifar10, get_driving_data_2
from torchsummary import summary
import torch
import matplotlib.pyplot as plt
import numpy as np

# Training Loop

NUM_EPOCHS = 2

model = DAVE2_D()
summary(model, (3, 66, 200))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# trainloader, testloader, classes = get_cifar10()
trainloader, testloader = get_driving_data_2()


for epoch in range(NUM_EPOCHS):
	# TRAIN
	temp_loss = 0.0
	for i, data in enumerate(tqdm(trainloader)):
		images, labels = data

		# zero the parameter gradients
		optimizer.zero_grad()

		out = model(images).squeeze(1)
		loss = criterion(out.float(), labels.float())
		loss.backward()
		optimizer.step()
		temp_loss += loss.item()
		if i % 2000 == 1999:
			print(f'[{epoch + 1}, {i + 1:5d}] loss: {temp_loss / 2000:.3f}')
			temp_loss = 0.0

	# TEST
	correct = 0
	total = 0

	with torch.no_grad():
		for i, data in enumerate(tqdm(testloader)):
			images, labels = data

			out = model(images).squeeze(1)
			_, predicted = torch.max(out.data, 1)
			total += labels.size(0)
			correct += (predicted==labels).sum().item()
	print(f'Accuracy of the network on the {len(testloader.dataset)} test images: {100 * correct // total} %')

print("Training complete")
