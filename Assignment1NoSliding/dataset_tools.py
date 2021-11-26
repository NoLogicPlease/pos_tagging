import re
import os
import zipfile

import numpy as np
import pandas as pd
from urllib import request
from tqdm import tqdm
from TwoWay import TwoWay


### CREATES THE DATASET


def download_dataset(download_path: str, url: str):  # download dataset
    if not os.path.exists(download_path):
        print("Downloading dataset...")
        request.urlretrieve(url, download_path)
        print("Download complete!")


def extract_dataset(download_path: str, extract_path: str):  # extract dataset
    print("Extracting dataset... (it may take a while...)")
    with zipfile.ZipFile(download_path, 'r') as loaded_zip:
        loaded_zip.extractall(extract_path)
    print("Extraction completed!")


def encode_dataset(dataset_path: str, dataset_folder: str, debug: bool = True) -> (
        pd.DataFrame, int):  # dataset to dataframe
    dataframe_rows = []
    s_lengths = []
    for index, file in tqdm(enumerate(sorted(os.listdir(dataset_path)))):
        file_name = os.path.join(dataset_path, file)
        with open(file_name) as f:
            lines = f.readlines()
        full_file = ''.join(lines)  # since lines is a list we use a single string called full_file
        full_file = re.sub(r'(\t\d+)', '', full_file)  # remove numbers from each lines of dataset
        full_file = re.sub(r'(\t)', ' ', full_file)  # replace \t with a space
        sentences = full_file.split('\n\n')
        for s in sentences:  # separate all words from their tags
            text = ''.join(re.findall(r'.+ ', s))
            labels = ''.join(re.findall(r' .+', s))
            labels = re.sub(r' (.+)', r'\1 ', labels)
            labels = re.sub('\n', ' ', labels)
            s_lengths.append(len(labels.split(' ')))
            # split into train, val and test
            if index <= 100:
                split = 'train'
            elif 100 < index <= 150:
                split = 'val'
            else:
                split = 'test'

            # create a single row of dataframe
            dataframe_row = {
                "text": text,
                "POStagging": labels,
                "split": split,
            }
            dataframe_rows.append(dataframe_row)

    # transform the list of rows in a proper dataframe
    df = pd.DataFrame(dataframe_rows)
    df = df[["text",
             "POStagging",
             "split"]]
    dataframe_path = os.path.join(dataset_folder, "dependency_treebank_df.pkl")
    df.to_pickle(dataframe_path)
    return df, max(s_lengths)


def create_dataset():
    dataset_folder = os.path.join(os.getcwd(), "Datasets")
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    url = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/dependency_treebank.zip"

    dataset_path_zip = os.path.join(dataset_folder, "dependency_treebank.zip")
    download_dataset(dataset_path_zip, url)
    extract_dataset(dataset_path_zip, dataset_folder)
    dataset_path = os.path.join(dataset_folder, "dependency_treebank")
    df, max_seq_len = encode_dataset(dataset_path, dataset_folder)
    df['POStagging'] = df['POStagging'].str.lower()
    df['text'] = df['text'].str.lower()
    return df, max_seq_len


def create_trainable(dataset, value_to_key, max_seq_len, num_classes):
    text_ids = [[value_to_key[word] for word in sen.split()] for sen in dataset['text']]
    x_train = []
    y_train = []
    label_tokenizer = {}

    one_hot_idx = 0
    for sen, tagging in zip(text_ids, dataset["POStagging"]):
        tmp = [0] * (max_seq_len - len(sen)) + sen
        x_train.append(tmp)

        for label in tagging.split():
            try:
                check_label = label_tokenizer[label]
            except KeyError:
                label_tokenizer[label] = [1 if i == one_hot_idx else 0 for i in range(num_classes)]
                one_hot_idx += 1

        tmp = [[0] * num_classes] * (max_seq_len - len(sen)) + [label_tokenizer[e] for e in tagging.split()]
        y_train.append(tmp)
    return np.array(x_train), np.array(y_train), label_tokenizer


def get_num_classes(dataset):
    return len(np.unique(''.join(dataset["POStagging"]).split())) + 1  # +1 for the padding
