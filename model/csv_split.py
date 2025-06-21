import pandas as pd
import os
import re

def split_csv_by_column(input_path, output_dir, entities, keywords, common_columns=None):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_path)

    if common_columns == None:   # no common column
        common_columns = []

    for ent in entities:
        first_joints_col = []
        last_joints_col = []
        for keyword in keywords:
            pattern = re.compile(fr'^{ent}_{keyword}(\d+)$')
            for col in  df.columns:
                m = pattern.match(col)
                if not m:
                    continue
                idx = int(m.group(1))
                if idx < 3:
                    first_joints_col.append(col)
                else:
                    last_joints_col.append(col)

        output_path = os.path.join(output_dir, ent)
        first_joints_df = df[common_columns + first_joints_col]
        first_joints_df.to_csv(output_path+'-FirstThreeJoints.csv')
        last_joints_df = df[common_columns + last_joints_col]
        last_joints_df.to_csv(output_path+'-LastThreeJoints.csv')
        print(f"Saved {ent} data.")

        


if __name__ == "__main__":
    dataset_root = "../../Dataset/"
    output_path = ["train_0620", "testing_0620"]
    input_path  = ["3286583.708339442-joint_data.csv",
                   "3288280.803231152-joint_data.csv"]
    entities = ['master1', 'master2', 'puppet']
    keywords = ['q', 'dq', 'tau']
    common_columns = ['timestamp']
    
    for index, input in enumerate(input_path):
        split_csv_by_column(input_path=(dataset_root+input), output_dir=(dataset_root+output_path[index]), entities=entities, keywords=keywords, common_columns=common_columns)
        print(f"Saved {output_path[index]}.")
        print("")