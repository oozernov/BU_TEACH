# NER with NLP Models

Detects NER in provided csv files using NLP models. Run by

```
python ner_check.py --csv_path ./sample_2.csv --ignore_cols [column name]
```


## Input

### csv_path

Path to standard CSV file with header as first row

### ignore_cols [Optional]

Name of columns to ignore when performing NER. Name must match with column name in csv, if ignoring multiple columns, seperate column names with comma `','` .


## Output

Outputs detection results as a `panda.DataFrame`

```
   col_name  row_index                                     content                                     type
0      name  [0, 1, 2]                    [[John], [Jill], [Jack]]           [[PERSON], [PERSON], [PERSON]]
1      city  [0, 1, 2]      [[New York], [Los Angeles], [Chicago]]     [[LOCATION], [LOCATION], [LOCATION]]
2  birthday  [0, 1, 2]  [[12/03/1992], [04/10/1983], [03/22/2002]]  [[DATE_TIME], [DATE_TIME], [DATE_TIME]]
```