"""
Data preprocessing script

Notes:
- This is a direct transformation of the provided notebook cells into a script.
- The script expects paths via `args.raw_xlsx` and `args.clinical.csv` exactly as written below.
args.xlsx should point to the path containing your raw data
args.clinical_csv should point to the path that your processed csv file is expected to be at. 
"""


import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def main(args):
    # ----------------------------
    # Load raw Excel
    # ----------------------------
    xls_path = Path(args.raw_xlsx)
    df = pd.read_excel(xls_path, sheet_name=0)
    df.head()  # preview (kept from notebook for parity)

    # ----------------------------
    # Missingness snapshot (initial)
    # ----------------------------
    missing_counts = df.isnull().sum() / df.shape[0] * 100

    # ----------------------------
    # Inspect categorical symptom-like columns (value counts)
    # ----------------------------
    cols = [
        'AXNAUSEA', 'AXVOMIT', 'AXDIARRH', 'AXCONSTP', 'AXABDOMN', 'AXSWEATN',
        'AXDIZZY', 'AXENERGY', 'AXDROWSY', 'AXVISION', 'AXHDACHE', 'AXDRYMTH',
        'AXBREATH', 'AXCOUGH', 'AXPALPIT', 'AXCHEST', 'AXURNDIS', 'AXURNFRQ',
        'AXANKLE', 'AXMUSCLE', 'AXRASH', 'AXINSOMN', 'AXDPMOOD', 'AXCRYING',
        'AXELMOOD', 'AXWANDER', 'AXFALL', 'AXOTHER'
    ]

    # ----------------------------
    # Drop the following columns with just 1 unique value
    # ----------------------------
    df = df.drop(columns=[
        'AXNAUSEA', 'AXVOMIT', 'AXDIARRH', 'AXCONSTP', 'AXABDOMN', 'AXSWEATN',
        'AXDIZZY', 'AXENERGY', 'AXDROWSY', 'AXVISION', 'AXHDACHE', 'AXDRYMTH',
        'AXBREATH', 'AXCOUGH', 'AXPALPIT', 'AXCHEST', 'AXURNDIS', 'AXURNFRQ',
        'AXANKLE', 'AXMUSCLE', 'AXRASH', 'AXINSOMN', 'AXDPMOOD', 'AXCRYING',
        'AXELMOOD', 'AXWANDER', 'AXFALL', 'AXOTHER'
    ])

    # ----------------------------
    # Drop MEMORY*/LANG* columns
    # ----------------------------
    df = df.drop(columns=[col for col in df.columns if col.startswith("MEMORY") or col.startswith("LANG")])


    # ----------------------------
    # Drop rows missing key vitals/targets (VSWEIGHT, then ADAS_TOTSCORE)
    # ----------------------------
    df = df.dropna(subset=['VSWEIGHT'])
    missing_counts = df.isnull().sum() / df.shape[0] * 100
    missing_counts[missing_counts > 0]

    df = df.dropna(subset=['ADAS_TOTSCORE'])
    missing_counts = df.isnull().sum() / df.shape[0] * 100
    missing_counts[missing_counts > 0]

    # ----------------------------
    # Unit normalization: weight → kg (VSWTUNIT==1 means lb), then drop unit
    # ----------------------------
    df.loc[df["VSWTUNIT"] == 1, "VSWEIGHT"] = df.loc[df["VSWTUNIT"] == 1, "VSWEIGHT"] * 0.453592
    df = df.drop(columns=["VSWTUNIT"])

    # ----------------------------
    # Unit normalization: height → cm (VSHTUNIT==1 means inches), then drop unit
    # ----------------------------
    df.loc[df["VSHTUNIT"] == 1, "VSHEIGHT"] = df.loc[df["VSHTUNIT"] == 1, "VSHEIGHT"] * 2.54
    df = df.drop(columns=["VSHTUNIT"])

    # ----------------------------
    # Clean/impute vitals (replace sentinels, then fill with column means rounded to 0.1)
    # ----------------------------
    cols_to_fix = ["VSWEIGHT", "VSHEIGHT", "VSBPSYS", "VSBPDIA", "VSPULSE", "VSRESP", "VSTEMP"]

    # Replace explicit sentinel values with NaN
    df[cols_to_fix] = df[cols_to_fix].replace(0.0, np.nan)
    df[cols_to_fix] = df[cols_to_fix].replace(-1, np.nan)

    # Impute using column means (rounded to 1 decimal place)
    df[cols_to_fix] = df[cols_to_fix].fillna(df[cols_to_fix].mean().round(1))

    # ----------------------------
    # Target relabel: DIAGNOSIS → {0,1,2} by (value - 1)
    # ----------------------------
    labs = pd.to_numeric(df["DIAGNOSIS"], errors="coerce")
    df["DIAGNOSIS"] = (labs - 1).astype(int)   # 1→0, 2→1, 3→2

    # ----------------------------
    # Drop unneeded column
    # ----------------------------
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    # ----------------------------
    # Save processed CSV
    # ----------------------------
    df.to_csv(args.clinical_csv)  # kept verbatim


parser = argparse.ArgumentParser()
parser.add_argument("--raw_xlsx",  type=str, default="/home/alaa.mohamed/ADNI_data/ADNI_Subset_Diagnosis_Filtered.xlsx")
parser.add_argument("--clinical_csv",  type=str, default="/home/alaa.mohamed/ADNI_MMF/data/clinical.csv")
args = parser.parse_args()

if __name__ == "__main__":
    main(args)

