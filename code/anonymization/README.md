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
python analyze_csv_spacy.py --csv_path ./ecri2022_mini.csv --ignore_cols 'StuID'
```

to analyze the CSV file and detect the locations of the inserted PII.