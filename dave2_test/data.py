#!/usr/bin/env python3

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

def get_cifar10():
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Resize((64,64)),
     	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     	])

	batch_size = 4

	trainset = torchvision.datasets.CIFAR10(root='/home/avishkar/Desktop/research', train=True,download=True, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=2)

	testset = torchvision.datasets.CIFAR10(root='/home/avishkar/Desktop/research', train=False,download=True, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=2)

	classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	return trainloader, testloader, classes


class DrivingDataset(Dataset):
	def __init__(self, labels_path, data_dir, transform=None):
		super().__init__()
		with open(Path(labels_path), "r") as f:
			self.labels = f.readlines()
			f.close()
		self.data_dir = Path(data_dir)
		self.transform = transform

	def __len__(self):
		return len(self.labels)
    
	def __getitem__(self, index) :
		img_path, label = self.labels[index].split()
		label=float(label)*np.pi/180
		img = Image.open(self.data_dir/img_path)
		width, height = img.size 
		img = img.crop((0, 150, width, height))
		# img = img.resize((66, 200))

		if self.transform:
			img = self.transform(img)

		return (img, label)

class DrivingDataset2(Dataset):
	def __init__(self, labels_path, data_dir, transform=None):
		super().__init__()
		with open(Path(labels_path), "r") as f:
			self.labels = f.readlines()
			f.close()
		self.data_dir = Path(data_dir)
		self.transform = transform

	def __len__(self):
		return(len(self.labels))

	def __getitem__(self, index):
		img_path, label, _ = self.labels[index].split()
		label = float(label.split(",")[0]) * np.pi/180

		img = Image.open(self.data_dir/img_path)
		width, height = img.size 
		img = img.crop((0, 140, width, height))
		img = img.resize((200, 66))
		
		if self.transform:
			img = self.transform(img)

		return (img, label)

class DrivingDataset_gemini(Dataset):
    """Custom dataset class for driving data."""

    def __init__(self, data_folder, train_file, limit=None, transform=None):
        """
        Args:
            data_folder (str): Path to the folder containing driving data.
            train_file (str): Path to the text file listing image paths and angles.
            limit (int, optional): Limit on the number of samples to load. Defaults to None (all data).
            transform (callable, optional): Transformation to apply to images. Defaults to None.
        """
        super().__init__()
        self.data_folder = data_folder
        self.train_file = train_file
        self.limit = limit
        self.transform = transform

        self.image_paths = []
        self.angles = []
        self.load_data()

    def load_data(self):
        """Loads image paths and angles from the text file."""
        with open(self.train_file) as fp:
            for line in islice(fp, self.limit):
                path, angle = line.strip().split()
                self.image_paths.append(os.path.join(self.data_folder, path))
                self.angles.append(float(angle) * np.pi / 180)  # Convert to radians

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves a sample (image and angle) from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the preprocessed image and the angle.
        """
        image_path = self.image_paths[idx]
        angle = self.angles[idx]

        # Read image using OpenCV (consider using other libraries if preferred)
        image = cv2.imread(image_path)

        # Preprocess: resize and convert to grayscale (HSV channel 1)
        image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:, :, 1], (100, 100))

        # Apply user-defined transformations (if provided)
        if self.transform:
            image = self.transform(image)

        # Convert to PyTorch tensor and normalize (consider using appropriate normalization for your task)
        image = torch.from_numpy(image).float() / 255.0

        return image, angle


def get_driving_data():
	transform = transforms.Compose([
	    transforms.ToTensor(),
	    # transforms.Resize((66,200)),
	    # transforms.RandomHorizontalFlip(),
	    # transforms.RandomRotation(5),
	    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	])


	BATCH_SIZE = 4
	NUM_WORKERS = 2
	DATA_DIR = Path("/home/avishkar/Desktop/research/driving_dataset")
	LABELS_PATH = DATA_DIR/"data.txt"

	# dataset = DrivingDataset(labels_path=LABELS_PATH, data_dir=DATA_DIR, transform=transform)
	dataset = DrivingDataset_gemini(DATA_DIR, LABELS_PATH, transform=transform)
	train_size = int(0.8 * len(dataset))
	test_size = int(len(dataset)-train_size)
	train_set, test_set = random_split(dataset, [train_size, test_size])
	train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,shuffle=True)
	test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

	return train_loader, test_loader


def get_driving_data_2():
	DATA_DIR = Path("/home/avishkar/Desktop/research/driving_dataset_2")
	LABELS_PATH = DATA_DIR/"data.txt"
	DATA_DIR = DATA_DIR/"data"

	transform = transforms.Compose([
	    transforms.ToTensor(),
	    # transforms.Resize((66,200)),
	    # transforms.RandomHorizontalFlip(),
	    # transforms.RandomRotation(5),
	    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	])

	BATCH_SIZE = 4
	NUM_WORKERS = 2

	dataset =  DrivingDataset2(LABELS_PATH, DATA_DIR, transform=transform)
	train_size = int(0.8 * len(dataset))
	test_size = int(len(dataset)-train_size)
	
	train_set, test_set = random_split(dataset, [train_size, test_size])
	train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,shuffle=True)
	test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

	return train_loader, test_loader	


def imshow(img):
	img = img / 2 + 0.5     # unnormalize
	img	 = img.numpy()
	plt.imshow(np.transpose(img, (1, 2, 0)))
	# plt.imshow(img)
	plt.show()

if __name__ == "__main__":
	# trainloader, testloader = get_driving_data()
	train_loader, test_loader = get_driving_data_2()
	print("Train len : ",len(train_loader.dataset), ", Test len : ",len(test_loader.dataset))

	dataiter = iter(train_loader)
	images, labels = next(dataiter)
	print("len labels ",len(labels))
	# print(images.shape)

	# DISPLAY IMAGES
	print(labels)
	imshow(torchvision.utils.make_grid(images, nrow =2))
	# print labels
	batch_size = 4





