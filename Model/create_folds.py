import os                                                                       #importing libraries
import pandas as pd
from sklearn import model_selection


if __name__ == "__main__":
    input_path = "D:/Atom_project/Melanoma/data_melanoma/"                      #assigning the path
    df = pd.read_csv(os.path.join(input_path, "train.csv" ))                    #reading the csv from the path
    df["kfold"] = -1                                                            #new folder 'kflod' and fill it with -1
    df = df.sample(frac=1).reset_index(drop=True)                               #shuffling the dataset
    y = df.target.values                                                        #retrieving the target values
    kf = model_selection.StratifiedKFold(n_splits=10)                           #using fold number 10
    for fold_, (_, _) in enumerate(kf.split(X=df, y=y)):
        df.loc[:, "kfold"] = fold_                                              #just adding the fold number to each fold
    df.to_csv(os.path.join(input_path, "train_folds.csv"), index = False)       #saving file as 'train_folds'

#run here and check the fold column!
