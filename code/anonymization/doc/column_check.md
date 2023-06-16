# Column Check

The scripts executes string matching on each column name to search for potential HIPPA data columns in dataset.

```
python column_check.py --csv_path ./sample_2.csv --custom_hippa_list ./example_hippa.txt
```

Currently the code checks for the following substrings, we can also specify custom lists through `--custom_hippa_list`.

```
['date', 'birth', 'day', 'age','time', 'name', 'address', 'city', 'id', 'fax',
'email', 'ssn', 'social', 'security', 'medical', 'liscence', 'tel', 'phone', 'number',
'account', 'ip', 'serial',]
```

## Input

### csv_path

Path to standard csv file with header as first row. 

### custom_hippa_list [Optional]

Path to text file with one word each line. e.g.

```
date
birth
age
...
```

## Output

Currently outputs all matched column names as a `List` and prints out to string. e.g.

```
['id', 'name', 'city', 'birthday', 'age']
```