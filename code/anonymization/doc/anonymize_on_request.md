# Anonymize on Request

We anonymize the provided csv file given user input. Run by
```
python anonymize_csv.py --csv_path ./sample_2.csv --request_path ./sample_request.json
```

## Inputs

### CSV File

Standard CSV file with header as first row.

### Request

User provides requests on what column to anonymize and how. Currently takes `.json` file in the format:

```
[
    // Column 1
    {
        "key": "name",      // column name: must match with csv columns
        "type": "NAME",     // column data type
        "mode": "RECODE"    // method used to anonymize
    },
    // Column 2
    {
        "key": "city",
        "type": "NAME",
        "mode": "REDACT"
    },  
    // Column 3
    {
        "key": "birthday",
        "type": "DATE",
        "mode": "RECODE",
        "format": "%m/%d/%Y"
    }, 
    // Column 4
    {
        "key": "age",
        "type": "AGE",
        "mode": "RECODE"
    }
    ...
]
```

#### key 

`key` specifies name of the column to be anonymized, must match the column name in the provied csv file.


#### type

Specifies the type of data to by anonymized, used to determine method used for recoding data. Currently supported types are: 
`NAME`, `DATE`, `AGE`.

- `NAME`: Expected data type string. Default mode: `RECODE`
- `DATE`: Expected data type string in the format compliant with [python datetime module](https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior). Default mode: `RECODE`
- `AGE`: Expected data type int. Default mode: `RECODE`

#### mode

Method used for anonymization, currently supports `REDACT`, `RECODE`.

- `REDACT`: Redacts all data, applies to all data type
- `RECODE`: Recodes data, actual behavior depends on datatype. Currently supports `NAME`, `DATE`, `AGE`, with behavior:
    - `NAME`: encodes string by hash
    - `DATE`: introduces random date perturbations to date ( +/- number of days sampled from uniform distribution with range [-200, +200]).  **Must include `format` field in request, specifying format used for date in complient with [python datetime module](https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior)**
    - `AGE`: maps age to a range.

## Output

Currently outputs `pandas.DataFrame`. e.g.

```
   id      name city    birthday    age
0   1  527bd5b5    *  01/12/1993  30-40
1   2  94232713    *  05/20/1983  40-50
2   3  4ff9fc6e    *  05/01/2002  20-30

```