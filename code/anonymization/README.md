# Data Anonymization


## Requirements

```
pip install presidio-analyzer
pip install presidio-anonymizer
python -m spacy download en_core_web_trf
```
## Column Name Check

Given a dataset, we check the column names to identify columns that may contain identifying information such as names, dates of birth, and addresses.

For example given a toy dataset `sample_2.csv`, 
```
   id  name         city     birthday  age
0   1  John     New York   12/03/1992   31
1   2  Jill  Los Angeles   04/10/1983   40
2   3  Jack      Chicago   03/22/2002   25
```

we run:
```
python column_check.py --csv_path ./sample_2.csv

# returns
['id', 'name', 'city', 'birthday', 'age']
```

## Anonymization on Request

Given user input (e.g. `sample_request.json`), specifying the column to anonymize, entry type, and mode (`REDACT` or `RECODE` , currently support type `NAME`, `DATE`, `AGE`), we anonymize by running.

```
python anonymize_csv.py --csv_path ./sample_2.csv --request_path ./sample_request.json
``` 
we get the anonymized dataset, where we hashed `name`, redacted `city`, added random perturbation to `birthday`, and replaced `age` with a range.

```
   id      name city    birthday    age
0   1  527bd5b5    *  01/12/1993  30-40
1   2  94232713    *  05/20/1983  40-50
2   3  4ff9fc6e    *  05/01/2002  20-30
```


## NLP NER Detection

To perform NER on dataset using LLM, run

```
python csv_check.py --csv_path ./sample_2.csv
```

to analyze the CSV file and detect the locations of the inserted PII. Running the above command should give the following output:

```
   col_name  row_index                                     content                                     type
0      name  [0, 1, 2]                    [[John], [Jill], [Jack]]           [[PERSON], [PERSON], [PERSON]]
1      city  [0, 1, 2]      [[New York], [Los Angeles], [Chicago]]     [[LOCATION], [LOCATION], [LOCATION]]
2  birthday  [0, 1, 2]  [[12/03/1992], [04/10/1983], [03/22/2002]]  [[DATE_TIME], [DATE_TIME], [DATE_TIME]]
```


