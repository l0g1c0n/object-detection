import time
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from PIL import Image, ImageDraw
from torchvision import transforms, datasets
import numpy as np
from scipy import ndimage

model_file = "crop_model2.pth"

image_size = 256
epochs = 1500
batch_size = 16
learning_rate =0.0001
dataset = "datatest"



def open_image_to_grayscale(image_path,image_size):
    # Open the image using PIL
    image = Image.open(image_path)

    image = image.resize((image_size,image_size))
    
    grayscale_image = image.convert('L')

    # Get the pixel data as a list
    pixel_list = list(grayscale_image.getdata())
    pixel_list = cut(pixel_list,image_size)
    return pixel_list

def cut(input_list, cutin):
    nested_ls = []
    ls = []
    for i in range(len(input_list)):
        if len(ls)+1 == cutin:
            ls.append(input_list[i])
            nested_ls.append(ls)
            ls = []
        else:
            ls.append(input_list[i])

    return nested_ls






def detect_black_pixels(pixel_list, min_size=10):
    labeled_array, num_features = ndimage.label(np.array(pixel_list) == 0)

    ls = []
    for label in range(1, num_features + 1):
        y, x = np.where(labeled_array == label)
        minx, miny = min(x), min(y)
        maxx, maxy = max(x), max(y)
        
        if (maxx - minx + 1) >= min_size and (maxy - miny + 1) >= min_size:
            ls.append((maxx, maxy, minx, miny))

    return ls



class PairedImageDataset(Dataset):
    def __init__(self, image_paths_list, transform=None):
        self.image_paths_list = image_paths_list
        self.transform = transform
        self.max_num = 0

    def __len__(self):
        return len(self.image_paths_list)

    def __getitem__(self, index):
        img_path, coords_img_path = self.image_paths_list[index]

        # Open the images using PIL
        image = Image.open(img_path)
        coords_image = Image.open(coords_img_path)
        image = image.convert("RGB")
        # Apply the transforms
        if self.transform is not None:
            image = self.transform[0](image)
            coords_image = self.transform[1](coords_image)

        # Get the bounding box coordinates
        ls : list[tuple] = detect_black_pixels(open_image_to_grayscale(coords_img_path,image_size))
        if self.max_num != 0:
            l = len(ls)
            for i in range(max_num - l):
                ls.append((0,0,0,0))
            

        
        
        # Return the paired images and bounding box coordinates
        return image, coords_image, ls

mean_rgb = (0.5, 0.5, 0.5)
std_rgb = (0.5, 0.5, 0.5)
mean_grayscale = (0.5,)
std_grayscale = (0.5,)
transform_rgb = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean_rgb, std_rgb)
])

transform_grayscale = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean_grayscale, std_grayscale)
])

images_dir = f"{dataset}/real"
box_dir = f"{dataset}/box"
image_paths_list= []
for i in range(len(os.listdir(images_dir))):
    real = os.listdir(images_dir)
    box = os.listdir(box_dir)
    
    pair = (os.path.join(images_dir,real[i]),os.path.join(box_dir,box[i]))
    image_paths_list.append(pair)



epochs = int(epochs / len(image_paths_list))





paired_dataset = PairedImageDataset(image_paths_list, transform=(transform_rgb,transform_grayscale))
data_loader = DataLoader(paired_dataset, batch_size=batch_size, shuffle=True, num_workers=0)


class Net(nn.Module):
    def __init__(self, num_channels=3, image_size= 64, num_output=4):
        super(Net, self).__init__()
        self.image_size = image_size
        self.conv1 = nn.Conv2d(num_channels, 16, 3, 1, 1)
        self.fc1 = nn.Linear(16 * (image_size // 4) * image_size, 128)
        self.fc2 = nn.Linear(128, num_output)  # Updated output layer for 4 coordinates

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 16 * (image_size // 4) * image_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Train the model
def train(model, data_loader, criterion, optimizer, num_epochs=30):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for _,(inputs,coords,labels) in enumerate(paired_dataset):
            
            optimizer.zero_grad()
            tensors = []
            for item in labels:
                
                t = torch.tensor(item, dtype=torch.float32)
                tensors.append(t)

            labels = torch.stack(tensors)
            labels = labels.view(-1,4)
            outputs = model(inputs)
            outputs = outputs.reshape(num_out, 4)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss = loss.item()

            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{_}/{len(paired_dataset)}]Loss: {running_loss:.4f}")
    torch.save(model.state_dict(), model_file)

max_num = 0
for item in paired_dataset:
    lbl = item[2]
    length = len(lbl)
    if length > max_num:
        max_num = length
paired_dataset.max_num = max_num
    
num_out = len(paired_dataset[0][2])

model = Net(num_output=num_out * 4,image_size = image_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)



if not os.path.isfile(model_file):
    train(model, data_loader, criterion, optimizer, num_epochs=epochs)
else:
    model.load_state_dict(torch.load(model_file))






def cut(input_list, cutin):
    nested_ls = []
    ls = []
    for i in range(len(input_list)):
        if len(ls)+1 == cutin:
            ls.append(input_list[i])
            nested_ls.append(ls)
            ls = []
        else:
            ls.append(input_list[i])

    return nested_ls


def predict_coords(image_path, model):
    image = Image.open(image_path)
    image = image.convert("RGB")
    image = transform_rgb(image).unsqueeze(0)
    outputs = model(image)
    
    outputs = [round(item) if item > 0 else 0 for item in outputs.tolist()[0]]
    outputs = cut(outputs,4)
    outputs = [item for item in outputs if not item.count(0) >= 2]
    
    return outputs


def draw_bounding_box(image, bbox):
    draw = ImageDraw.Draw(image)
    for coords in bbox:
        x, y, width, height = coords
        draw.rectangle((x, y, width , height), outline="lime", width=2)


# Evaluation loop
while True:
    image_path = input("Enter the path of the image (or 'exit' to quit): ")
    if image_path.lower() == 'exit':
        break

    start = time.time()
    bbox_coords = predict_coords(image_path, model)
    time_to_predict = time.time() - start
    print(bbox_coords)

    image = Image.open(image_path)
    image = image.resize((image_size, image_size))
    draw_bounding_box(image, bbox_coords)
    image.show()

    time_to_show = time.time() - start

    print("time taken to predict: ", time_to_predict)
    print("time taken to predict and show: ", time_to_show)
    print("\n")
