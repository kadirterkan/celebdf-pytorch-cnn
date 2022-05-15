import pandas as pd

columns = ["anchor_index", "positive_index", "negative_index", "label", "test"]

def make_triplet(csv_path):

    df = pd.read_csv(csv_path)

    triplet_df = pd.DataFrame(columns=columns)

    for index, row in df.iterrows():
        video_class = row['class']
        original_face = row['original_face']

        if (video_class == 'real'):
            positive_index = df.loc[(df['class'] == 'real') & (df['original_face'] == original_face)].sample(random_state=1).index

            while (positive_index == index):
                print("loop?", index, positive_index)
                positive_index = df.loc[(df['class'] == 'real') & (df['original_face'] == original_face)].sample().index
                print("index changed?", positive_index)
            
            negative_index = df.loc[(df['class'] == 'fake') & (df['original_face'] == original_face)].sample(random_state=1).index
        else:
            positive_index = df.loc[(df['class'] == 'fake') & (df['original_face'] == original_face)].sample(random_state=1).index

            while (negative_index == index):
                print("loop?", index, positive_index)
                positive_index = df.loc[(df['class'] == 'fake') & (df['original_face'] == original_face)].sample().index
                print("index changed?", positive_index)
            
            negative_index = df.loc[(df['class'] == 'real') & (df['original_face'] == original_face)].sample(random_state=1).index
        
        triplet_df.loc[triplet_df.size+1] = [index, positive_index, negative_index, row['class'], row['test']]
    
    triplet_df.to_csv("triplet.csv")

make_triplet("result.csv")