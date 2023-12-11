import argparse
import numpy as np
import pandas as pd
import re
import string

HIPPA_LIST = ['date', 'birth', 'day', 'age', 'time', 'name', 'address', 'city', 'id', 'fax',
              'email', 'ssn', 'social', 'security', 'medical', 'liscence', 'tel', 'phone', 'number',
              'account', 'ip', 'serial',]

def text_preprocess(text):
    text = text.lower()  # lower case
    text = text.strip()  # extra space
    text = re.sub(' +', ' ', text) # remove extra space
    text = text.translate(str.maketrans('', '', string.punctuation)) # strip punctuations
    return text

def is_hippa(text, hippa_list, df=None):
    if text == 'age' and df is not None:
        # Check if any values in the 'age' column are greater than 80
        return df[text].apply(pd.to_numeric, errors='coerce').gt(80).any()
    return text in hippa_list

def check_columns(csv_path, hippa_list):
    df = pd.read_csv(csv_path, delimiter=',', header=0)
    hippa_cols = []
    column_names = df.columns
    for idx, col in enumerate(column_names.values):
        text = text_preprocess(col)
        if is_hippa(text, hippa_list, df if text == 'age' else None):
            hippa_cols.append(col)
    return hippa_cols

def get_parser():
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--csv_path', default=None, type=str, help='Path to CSV file.')
    parser.add_argument('--custom_hippa_list', default=None, type=str, help='Path to custom HIPPA name file.')
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    csv_path = args.csv_path

    custom_hippa_list = None
    if args.custom_hippa_list:
        with open(args.custom_hippa_list, 'r') as f:
            custom_hippa_list = [line.rstrip().lower() for line in f]

    cols = check_columns(csv_path, hippa_list= custom_hippa_list if custom_hippa_list else HIPPA_LIST)
    print('Potential HIPPA columns:', cols)
