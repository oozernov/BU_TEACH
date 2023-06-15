import numpy as np
import pandas as pd
import argparse
import json
import re
import random
from hashlib import md5
from datetime import datetime, timedelta

from column_check import text_preprocess

random.seed(2024)


preprocess = {
    'NAME': lambda x: text_preprocess(x),
    'DATE': lambda x: x.strip(),
    'AGE': lambda x: int(x) if type(x) != int else x
}


def redact_col(df, request):
    key = request['key']
    assert key in df.columns, "Column name {} does not exist".format(key)
    return df[key].apply(lambda x: '*')


def recode_col(df, request, MAX=200, MIN=-200):
    key = request['key']
    type = request['type']
    assert key in df.columns, "Column name {} does not exist".format(key)

    recoded_col = df[key].copy()
    

    if type == 'NAME':
        # map Name to ID
        recoded_col = recoded_col.apply(preprocess[type])
        recoded_col = recoded_col.apply(lambda x: md5(x.encode()).hexdigest()[:8])
    elif type == 'AGE':
        # map Age to decade
        recoded_col = recoded_col.apply(preprocess[type])
        recoded_col = recoded_col.apply(lambda x: "-".join([str((x//10)*10), str((x//10+1)*10)]))
        
    elif type == 'DATE':
        # random shift to Date (given date format)
        assert 'format' in request, "Missing date format"
        recoded_col = recoded_col.apply(preprocess[type])
        format = request['format']
        rand_days = random.randint(MIN, MAX)
        func_date_perturbe = lambda x: (datetime.strptime(x, format) + \
                                        timedelta(days=rand_days)).strftime(format)
        recoded_col = recoded_col.apply(func_date_perturbe)

    else:
        raise NotImplementedError

    return recoded_col


def anonymize_csv(csv_path, request_path):
    df = pd.read_csv(csv_path, delimiter=',', header=0)
    with open(request_path, 'r') as f:
        requests = json.load(f)

    for request in requests:
        if request['mode'] == 'REDACT':
            df[request['key']] = redact_col(df, request)
        elif request['mode'] == 'RECODE':
            df[request['key']] = recode_col(df, request)

    return df


def get_parser():
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--csv_path', default=None, type=str, help='Path to CSV file.')
    parser.add_argument('--request_path', default=None, type=str, help='path to anonymize request file')
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    result_pd = anonymize_csv(args.csv_path, args.request_path)
    print(result_pd)

