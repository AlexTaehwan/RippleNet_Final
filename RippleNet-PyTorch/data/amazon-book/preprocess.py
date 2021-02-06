import pandas as pd
import numpy as np
import json


def save_new_kg_relation_mapped(kg_file):
    data = pd.read_csv(kg_file, sep=" ", header=None)
    np.savetxt('kg_final.txt', data.values, fmt="%d\t%s\t%d")


def process_train(train, test):
    print("start processing")
    columns = ["source", "relation", "book_rating"]
    df = pd.DataFrame(columns=columns) 
    with open(train, "r") as f:
        for line in f:
            arr = line.strip().split()
            for i in range(1, len(arr)):
                new_row = {'source' : int(arr[0]), 'relation': int(arr[i]), "book_rating": 0}
                df = df.append(new_row, ignore_index=True)

    seperator = {'source' : 1, 'relation': 1, "book_rating": 1}
    df = df.append(seperator, ignore_index=True)
    with open(test, "r") as f:
        for line in f:
            arr = line.strip().split()
            for i in range(1, len(arr)):
                new_row = {'source' : int(arr[0]), 'relation': int(arr[i]), "book_rating": 0}
                df = df.append(new_row, ignore_index=True)
    np.savetxt('ratings_final.txt', df.values, fmt="%d\t%d\t%d")
    print("fininsh processing")
    

save_new_kg_relation_mapped("kg_final.txt")
process_train("train.txt", "test.txt")