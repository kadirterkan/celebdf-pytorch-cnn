import os
import torch.utils.data
import torchvision.transforms as transforms

import cv2 as cv
import pandas as pd
from PIL import Image

from extract_frames import get_dataframe

class TripletLoader(torch.util.data.Dataset):
    def __init__(self, triplet_df, frame_df, transform=None):
        self.triplet_df = triplet_df
        self.frame_df = frame_df
        self.transform = transform
    
    def __len__(self):
        return len(self.triplet_df)
    
    def __getitem__(self, index):
        anch_row = self.frame_df.iloc[self.triplet_df[index]["anchor_index"]]
        pos_row = self.frame_df.iloc[self.triplet_df[index]["positive_index"]]
        neg_row = self.frame_df.iloc[self.triplet_df[index]["negative_index"]]

        anch_path = anch_row["path"]
        pos_path = pos_row["path"]
        neg_path = neg_row["path"]

        anch = cv.imread(anch_path)
        post = cv.imread(pos_path)
        neg = cv.imread(neg_path)

        anch_box = [anch_row["x1"], anch_row["y1"], anch_row["x2"], anch_row["y2"]]
        post_box = [pos_row["x1"], pos_row["y1"], pos_row["x2"], pos_row["y2"]]
        neg_box = [neg_row["x1"], neg_row["y1"], neg_row["x2"], neg_row["y2"]]

        anch_face = self.crop_face(anch, anch_box)
        post_face = self.crop_face(post, post_box)
        neg_face = self.crop_face(neg, neg_box)

        if self.transform is not None:
            anch_face = self.transform(anch_face)
            post_face = self.transform(post_face)
            neg_face = self.transform(neg_face)

        return anch_face, post_face, neg_face
    
    def crop_face(img, box, margin=1):
        x1, y1, x2, y2 = box
        size = int(max(x2-x1, y2-y1) * margin)
        center_x, center_y = (x1 + x2)//2, (y1 + y2)//2
        x1, x2 = center_x-size//2, center_x+size//2
        y1, y2 = center_y-size//2, center_y+size//2
        face = Image.fromarray(img).crop([x1, y1, x2, y2])
        face = cv.cvtColor(face, cv.COLOR_RGB2BGR)
        return cv.resize(face, (160,160))

columns = ["path", "frame_name", "video_name", "frame_number", "class", "test", "original_video", "original_face", "target_face", "x1", "y1", "x2", "y2"]

def make_dataset_ready(path, extract_location):
    assert path

    df = pd.DataFrame(columns=columns)

    with open (os.path.join(path, "List_of_testing_videos.txt")) as f:
        lines = f.readLines()

    test_video_names = [(line.split(' ')[1]).split('/')[1].strip() for line in lines]

    get_dataframe(df, path, extract_location, test_video_names)

def get_triplet_dataset():

    transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    frame_df = pd.read_csv('frames.csv')
    triplet_df = pd.read_csv('triplet.csv')

    test_triplet_df = triplet_df.loc[triplet_df['test'] == True]
    train_triplet_df = triplet_df.loc[triplet_df['test'] == False]

    test_dataset = TripletLoader(test_triplet_df, frame_df, transform)
    train_dataset = TripletLoader(train_triplet_df, frame_df, transform)

    return test_dataset, train_dataset