import os
import cv2 as cv
import pandas as pd
import numpy as np
from facenet_pytorch import MTCNN

file_switcher = {
    'Celeb-real' : 'real',
    'Celeb-synthesis' : 'fake',
    'YouTube-real' : 'real'
}

df = None
path = None
extract_location = None
test_video_names = None
global_count = 0

mtcnn = MTCNN()

def get_dataframe(idf, ipath, iextract_location, itest_video_names):
    global df, path, extract_location, test_video_names

    df = idf
    path = ipath
    extract_location = iextract_location
    test_video_names = itest_video_names

    if not os.path.exists(extract_location):
        os.makedirs(extract_location)

    for directory_name in file_switcher:
        sub_directory = os.path.join(path, directory_name)
        open_directory(sub_directory, directory_name)
    
    df.to_csv("result.csv")

def open_directory(sub_path, directory_name):
    
    for file in os.listdir(sub_path):
        is_test = False

        if file in test_video_names:
            is_test = True
        
        extract_frames(os.path.join(sub_path, file), file, directory_name, is_test)

def extract_frames(file_path, file, directory_name, is_test):

    test_train = "train"
    if is_test:
        test_train = "test"
    extract_path = os.path.join(extract_location, test_train, directory_name, file)

    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
    
    cap = cv.VideoCapture(file_path)
    length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv.CAP_PROP_FPS)
    step = int(length / fps)
    frame_count = 0

    video_label = file_switcher.get(directory_name)

    if (video_label == 'real'):
        video_name = file
        file_name = (file.split('.')[0]).split('_')
        original_face = file_name[0]
        original_video = None
        target_face = None
    else:
        video_name = file
        file_name = (file.split('.')[0]).split('_')
        original_face = file_name[0]
        video_no = file_name[2]
        original_video = original_face + "_" + video_no
        target_face = file_name[1]


    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Stream has ended. Exiting video " + file)
            break

        img_np = np.array(frame)
        frame = cv.cvtColor(img_np, cv.COLOR_RGB2BGR)

        box = face_detect(frame)

        if (box is None):
            frame_count += step
            cap.set(cv.CAP_PROP_POS_FRAMES, frame_count)
            continue
        else:
            x1, y1, x2, y2 = box

            filename = "frame_number" + "_" + str(frame_count).zfill(4) + ".jpg"

            frame_path = os.path.join(extract_path, filename)

            df.loc[df.size+1] = [frame_path, filename, video_name, str(frame_count), video_label, is_test, original_video, original_face, target_face, x1, y1, x2, y2]
            cv.imwrite(frame_path, img_np)

            frame_count += step
            cap.set(cv.CAP_PROP_POS_FRAMES, frame_count)

    cap.release()
    cv.destroyAllWindows()

def face_detect(frame):
    results = mtcnn.detect(frame)
    
    if results is None:
        return None
    
    boxes, probs = results
    
    if boxes is None or probs is None:
        return None
    
    for j, box in enumerate(boxes):
        if probs[j] > 0.98:
            return box
        else:
            return None

columns = ["path", "frame_name", "video_name", "frame_number", "class", "test", "original_video", "original_face", "target_face", "x1", "y1", "x2", "y2"]
df = pd.DataFrame(columns=columns)

with open (os.path.join("/Users/kadirerkan/VSCode/celebdf/Celeb-DF-v2", "List_of_testing_videos.txt")) as f:
    lines = f.readlines()

test_video_names = [(line.split(' ')[1]).split('/')[1].strip() for line in lines]

get_dataframe(df, "/Users/kadirerkan/VSCode/celebdf/Celeb-DF-v2", "/Users/kadirerkan/VSCode/celebdf/extract", test_video_names)