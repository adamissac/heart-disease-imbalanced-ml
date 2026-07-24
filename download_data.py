"""
Download the three public datasets used in this study.

    python download_data.py

Files are written to data/ and are gitignored, since they total ~51 MB.
All three are public, de-identified datasets. Original sources are on Kaggle
(links below); the URLs here point to stable public mirrors so the download
needs no Kaggle account or API key.
"""
import os
import sys
import urllib.request

DATASETS = [
    {
        "name": "heart_2020_cleaned.csv",
        "desc": "CDC BRFSS 2020 heart disease (319,795 records) - primary dataset",
        "url": "https://raw.githubusercontent.com/Sylv94/heart_2020_cleaned/main/heart_2020_cleaned.csv",
        "source": "https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease",
        "expect_rows": 319795,
    },
    {
        "name": "cardio_train.csv",
        "desc": "Cardiovascular disease, clinical exams (70,000 records) - continuous-feature replication",
        "url": "https://raw.githubusercontent.com/krisadell01/cardiovascular_dataset/main/cardio_train.csv",
        "source": "https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset",
        "expect_rows": 70000,
    },
    {
        "name": "diabetes_binary_BRFSS2015.csv",
        "desc": "BRFSS 2015 diabetes health indicators (253,680 records) - binary-feature replication",
        "url": "https://raw.githubusercontent.com/pulawx-png/diabetes_binary_health_indicators_BRFSS2015/main/diabetes_binary_health_indicators_BRFSS2015.csv",
        "source": "https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset",
        "expect_rows": 253680,
    },
]


def main():
    os.makedirs("data", exist_ok=True)
    for d in DATASETS:
        path = os.path.join("data", d["name"])
        if os.path.exists(path):
            print(f"[skip] {d['name']} already present")
            continue
        print(f"[get ] {d['name']}  ({d['desc']})")
        try:
            urllib.request.urlretrieve(d["url"], path)
        except Exception as exc:
            print(f"       FAILED: {exc}")
            print(f"       Download manually from: {d['source']}")
            continue
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            n = sum(1 for _ in fh) - 1
        ok = "OK" if n == d["expect_rows"] else f"WARNING expected {d['expect_rows']}"
        print(f"       {n:,} rows  {ok}")
    print("\nDone. Data written to data/")


if __name__ == "__main__":
    sys.exit(main())
