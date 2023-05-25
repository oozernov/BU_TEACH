# Data Anonymization


## Requirements

```
pip install presidio-analyzer
pip install presidio-anonymizer
python -m spacy download en_core_web_trf
```


## Detection

The dataset `ecri2022_mini.csv` is a subset of the original ecri2022 dataset. PII such as names, addresses, and dates are randomly inserted to the dataset. Run

```
python csv_check.py --csv_path ./ecri2022_mini.csv --ignore_cols 'StuID'
```

to analyze the CSV file and detect the locations of the inserted PII. Running the above command should give the following output:

```
Analyzing...
Column Name: TeachID, Indexes: [262]
Column Name: Tx, Indexes: [197]
Column Name: ORFwc, Indexes: [278]
Column Name: na, Indexes: [131, 187]
Done
```

This indicates there are potential PII in row 262 (0 indexed) of column `TeacherID`, row 197 of column `Tx`, row 278 of column `ORFwc` and row 137, 187 of column `na`.

