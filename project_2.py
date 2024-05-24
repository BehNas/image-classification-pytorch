import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ImageLoader(Dataset):
    def __init__(self, rootPath, csv_path, transform = None):
        self.main_dataframe = pd.read_csv(csv_path)
        self.rootpath = rootPath
        self.transform = transform

    def __len__(self):
        return len(self.main_dataframe)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.rootpath, self.main_dataframe.iloc[index,0])
        image = Image.open(image_path).convert('RGB')
        label = int(self.main_dataframe.iloc[index, 1])
        if self.transform:
            image = self.transform(image)
    
        if len(self.main_dataframe.columns) > 1:  # Check if there are labels
                label = self.main_dataframe.iloc[index, 1]
                return image, label
        else:
                return image

batch_size=1
num_workers=0
rootPath_train = "/Users/behnaz/cloud-type-classification2/images/train"
rootPath_test = "/Users/behnaz/cloud-type-classification2/images/test"
csv_train_path = "/Users/behnaz/cloud-type-classification2/train.csv"
csv_test_path = "/Users/behnaz/cloud-type-classification2/test.csv"

train_transforms = transforms.Compose([transforms.ToTensor(), 
                                       transforms.Resize((224,224)),
                                    #    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                       ])
test_transforms = transforms.Compose([transforms.ToTensor(), 
                                      transforms.Resize((224,224)), 
                                    #   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                      ])
dataset_train = ImageLoader(rootPath_train, csv_train_path, transform=train_transforms)
dataset_test = ImageLoader(rootPath_test, csv_test_path, transform=test_transforms)

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

image, label = next(iter(train_loader))
print(image.shape)
image = image.squeeze().permute(1,2,0)
print(image.shape)
plt.imshow(image)
plt.show()
