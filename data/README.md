# Data

The three datasets are not committed to this repository (~51 MB total). Fetch them with:

```bash
python download_data.py
```

| File | Records | Description |
|---|---|---|
| `heart_2020_cleaned.csv` | 319,795 | CDC BRFSS 2020 heart disease. Primary dataset for the prevalence sweep. |
| `cardio_train.csv` | 70,000 | Cardiovascular disease from clinical exams. Continuous-feature replication. |
| `diabetes_binary_BRFSS2015.csv` | 253,680 | BRFSS 2015 diabetes indicators. Binary-feature replication. |

All three are public, de-identified datasets. Original Kaggle sources are listed in
`download_data.py` and in the main README.
