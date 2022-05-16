import pandas as pd

columns = ["anchor_index", "positive_index", "negative_index", "label", "test"]

def make_triplet(csv_path):

    df = pd.read_csv(csv_path)

    triplet_df = pd.DataFrame(columns=columns)

    for index, row in df.iterrows():
        print("Writing index ",index)
        video_class = row['class']
        original_face = row['original_face']

        print(row)

        if (video_class == 'real'):
            positive_rows = df.loc[(df['class'] == 'real') & (df['original_face'] == original_face)]

            if positive_rows.size != 0:
                positive_index = positive_rows.sample().index

                while (positive_index == index):
                    positive_index = positive_rows.sample().index
                
                negative_rows = df.loc[(df['class'] == 'fake') & (df['original_face'] == original_face)]

                if negative_rows.size != 0:
                    negative_index = negative_rows.sample().index
                else:
                    negative_rows = df.loc[(df['class'] == 'fake') & (df['target_face'] == original_face)]

                    if negative_rows.size != 0:
                        negative_index = negative_rows.sample().index
                    else:
                        negative_index = df.loc[(df['class'] == 'fake')].sample().index
            else:
                positive_index = df.loc[(df['class'] == 'real')].sample().index

                while (positive_index == index):
                    positive_index = df.loc[(df['class'] == 'real')].sample().index
                
                negative_index = df.loc[(df['class'] == 'fake')].sample().index
        else:
            positive_rows = df.loc[(df['class'] == 'fake') & (df['original_face'] == original_face)]
            
            if positive_rows.size != 0:
                positive_index = df.loc[(df['class'] == 'fake') & (df['original_face'] == original_face)]

                positive_index = df.loc[(df['class'] == 'fake') & (df['original_face'] == original_face)].sample().index

                while (positive_index == index):
                    positive_index = df.loc[(df['class'] == 'fake') & (df['original_face'] == original_face)].sample().index
                
                negative_rows = df.loc[(df['class'] == 'real') & (df['original_face'] == original_face)]

                if negative_rows.size != 0:
                    negative_index = negative_rows.sample().index
                else:
                    negative_index = df.loc[(df['class'] == 'real')].sample().index
            else:
                positive_rows = df.loc[(df['class'] == 'fake') & df['target_face'] == original_face]

                if positive_rows.size != 0:
                    positive_index = positive_rows.sample().index

                    while (positive_index == index):
                        positive_index = positive_rows.sample().index
                else:
                    positive_index = df.loc[(df['class'] == 'fake')].sample().index

                    while (positive_index == index):
                        positive_index = df.loc[(df['class'] == 'fake')].sample().index
                
                negative_index = df.loc[(df['class'] == 'real')].sample().index
    
        triplet_df.loc[triplet_df.size+1] = [index, positive_index, negative_index, row['class'], row['test']]
    
    triplet_df.to_csv("triplet.csv")

make_triplet("result.csv")