import h2o
import json
import pandas as pd

def separate_id_col(h2o_frame):
    possible_id_list = ['ID', 'Id', 'id']
    for i in possible_id_list:
        if i in h2o_frame.names:
            id_name = i
            X_id = h2o_frame[:, id_name]
            X_h2o = h2o_frame.drop(id_name)
            break
    else:
        id_name, X_id = None, None
        X_h2o = h2o_frame
    return id_name, X_id, X_h2o

def match_col_types(h2o_frame):
    with open('data/processed/train_col_types.json') as f:
        train_col_types = json.load(f)
    for key in train_col_types.keys():
        try:
            if train_col_types[key] != h2o_frame.types[key]:
                if train_col_types[key] == 'real' and h2o_frame.types[key] == 'enum':
                    h2o_frame[key] = h2o_frame[key].ascharacter().asnumeric()
                elif train_col_types[key] == 'real':
                    h2o_frame[key] = h2o_frame[key].asnumeric()
                elif train_col_types[key] == 'int':
                    h2o_frame[key] = h2o_frame[key].asfactor()
                elif train_col_types[key] == 'str':
                    h2o_frame[key] = h2o_frame[key].ascharacter()
        except:
            pass
    return h2o_frame

def preprocess_for_model(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
    df["Vehicle_Damage"] = df["Vehicle_Damage"].map({"Yes": 1, "No": 0})
    df["Vehicle_Age_lt_1Y"] = (df["Vehicle_Age"] == "< 1 Year").astype(int)
    df["Vehicle_Age_1_2Y"] = (df["Vehicle_Age"] == "1-2 Year").astype(int)
    df["Vehicle_Age_gt_2Y"] = (df["Vehicle_Age"] == "> 2 Years").astype(int)
    df.drop("Vehicle_Age", axis=1, inplace=True)
    df = pd.get_dummies(df, columns=["Region_Code", "Policy_Sales_Channel"],
                        prefix=["Region_Code", "Policy_Sales_Channel"])
    return df
