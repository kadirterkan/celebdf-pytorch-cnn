import os
from turtle import pd
import torch.utils.data as data
import torchvision.transforms as transforms
import torch

from sklearn.preprocessing import LabelEncoder
import cv2 as cv
import pandas as pd
from PIL import Image

lb = LabelEncoder()

class BaseLoader(data.Dataset):
    def __init__(self, frame_df, transform=None):
        frame_df['encoded_labels'] = lb.fit_transform(frame_df['class'])
        self.frame_df = frame_df
        self.transform = transform
    
    def __len__(self):
        return len(self.frame_df)
    
    def __getitem__(self, index):
        frame_row = self.frame_df.iloc[index]

        frame_path = frame_row["path"]

        frame = cv.imread(frame_path)

        frame_box = [frame_row["x1"], frame_row["y1"], frame_row["x2"], frame_row["y2"]]

        frame_face = self.crop_face(frame, frame_box)

        if self.transform is not None:
            frame_face = self.transform(frame_face)
        
        label = torch.tensor(self.frame_df.iloc[index]['encoded_labels'], dtype=torch.long)

        return frame_face, label

    def crop_face(img, box, margin=1):
        x1, y1, x2, y2 = box
        size = int(max(x2-x1, y2-y1) * margin)
        center_x, center_y = (x1 + x2)//2, (y1 + y2)//2
        x1, x2 = center_x-size//2, center_x+size//2
        y1, y2 = center_y-size//2, center_y+size//2
        face = Image.fromarray(img).crop([x1, y1, x2, y2])
        face = cv.cvtColor(face, cv.COLOR_RGB2BGR)
        return cv.resize(face, (224,224))

def get_base_dataset():

    transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    frame_df = pd.read_csv('frames.csv')

    test_frame_df = frame_df.loc[frame_df['test'] == True]
    train_frame_df = frame_df.loc[frame_df['test'] == False]

    test_dataset = BaseLoader(test_frame_df, transform)
    train_dataset = BaseLoader(train_frame_df, transform)

    return test_dataset, train_dataset