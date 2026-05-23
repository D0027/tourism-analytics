"""
Tourism Experience Analytics
src/data_pipeline.py — Load, clean, merge, and feature-engineer all datasets.
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────
# 1.  LOAD
# ─────────────────────────────────────────────────────────────────

def load_datasets(base_path: str) -> dict:
    """Read all Excel files from base_path and return a dict of DataFrames."""
    file_map = {
        "transaction": "Transaction.xlsx",
        "user":        "User.xlsx",
        "city":        "City.xlsx",
        "item":        "Updated_Item.xlsx",
        "mode":        "Mode.xlsx",
        "attr_type":   "Type.xlsx",
        "country":     "Country.xlsx",
        "region":      "Region.xlsx",
        "continent":   "Continent.xlsx",
    }
    datasets = {}
    print("Loading datasets …")
    for key, filename in file_map.items():
        path = os.path.join(base_path, filename)
        try:
            df = pd.read_excel(path)
            print(f"  ✔  {filename:<30}  {df.shape}")
            datasets[key] = df
        except Exception as e:
            print(f"  ✘  {filename}: {e}")
            datasets[key] = pd.DataFrame()
    print("Done.\n")
    return datasets


# ─────────────────────────────────────────────────────────────────
# 2.  CLEAN
# ─────────────────────────────────────────────────────────────────

def clean_datasets(ds: dict) -> dict:
    """In-place cleaning of each table; returns the same dict."""
    transaction = ds["transaction"]
    user        = ds["user"]
    city        = ds["city"]
    item        = ds["item"]
    mode        = ds["mode"]
    attr_type   = ds["attr_type"]
    country     = ds["country"]
    region      = ds["region"]
    continent   = ds["continent"]

    # ── Transaction ──────────────────────────────────────────────
    print("=== Transaction Cleaning ===")
    before = len(transaction)
    transaction.drop_duplicates(inplace=True)
    print(f"  Duplicates removed: {before - len(transaction)}")

    transaction["Rating"] = pd.to_numeric(transaction["Rating"], errors="coerce")
    transaction = transaction[transaction["Rating"].between(1, 5)]
    transaction.dropna(subset=["UserId", "AttractionId", "Rating", "VisitMode"], inplace=True)

    transaction["VisitMode"] = transaction["VisitMode"].astype(str).str.strip().str.title()
    transaction["VisitYear"]  = pd.to_numeric(transaction["VisitYear"],  errors="coerce")
    transaction["VisitMonth"] = pd.to_numeric(transaction["VisitMonth"], errors="coerce")
    transaction.dropna(subset=["VisitYear", "VisitMonth"], inplace=True)
    transaction[["VisitYear", "VisitMonth"]] = transaction[["VisitYear", "VisitMonth"]].astype(int)
    print(f"  Final Transaction shape: {transaction.shape}\n")

    # ── User ─────────────────────────────────────────────────────
    print("=== User Cleaning ===")
    user.drop_duplicates(subset=["UserId"], inplace=True)
    user.dropna(subset=["UserId", "ContinentId", "CountryId"], inplace=True)
    print(f"  User shape: {user.shape}\n")

    # ── Lookup tables ────────────────────────────────────────────
    for name, df in [("City", city), ("Item", item), ("Mode", mode),
                     ("Type", attr_type), ("Country", country),
                     ("Region", region), ("Continent", continent)]:
        df.drop_duplicates(inplace=True)
        df.dropna(how="all", inplace=True)
        print(f"  {name}: {df.shape}")

    # Standardise text
    city["CityName"]            = city["CityName"].str.strip().str.title()
    attr_type["AttractionType"] = attr_type["AttractionType"].str.strip().str.title()
    continent["Continent"]      = continent["Continent"].str.strip().str.title()
    country["Country"]          = country["Country"].str.strip().str.title()
    region["Region"]            = region["Region"].str.strip().str.title()
    item["Attraction"]          = item["Attraction"].str.strip().str.title()

    # Standardise Mode table name column
    mode_name_col = mode.columns[1]
    mode[mode_name_col] = mode[mode_name_col].str.strip().str.title()

    ds["transaction"] = transaction
    return ds


# ─────────────────────────────────────────────────────────────────
# 3.  MERGE
# ─────────────────────────────────────────────────────────────────

def build_master(ds: dict) -> pd.DataFrame:
    """Join all tables into one master DataFrame."""
    transaction = ds["transaction"]
    user        = ds["user"]
    city        = ds["city"]
    item        = ds["item"]
    attr_type   = ds["attr_type"]
    continent   = ds["continent"]
    region      = ds["region"]
    country     = ds["country"]
    mode_df     = ds["mode"]

    df = transaction.merge(user, on="UserId", how="left")
    print("Columns after transaction+user:", df.columns.tolist())

    df = df.rename(columns={"CityId": "CityId_user"})

    df = df.merge(
        city.rename(columns={"CityId": "CityId_user", "CityName": "UserCity",
                              "CountryId": "CountryId_user"}),
        on="CityId_user", how="left"
    )
    df = df.merge(continent, on="ContinentId", how="left")
    df = df.merge(region[["RegionId", "Region"]], on="RegionId", how="left")
    df = df.merge(country[["CountryId", "Country"]], on="CountryId", how="left")
    df = df.merge(item, on="AttractionId", how="left")
    df = df.merge(attr_type, on="AttractionTypeId", how="left")

    if "AttractionCityId" in df.columns:
        df = df.merge(
            city.rename(columns={"CityId": "AttractionCityId", "CityName": "AttractionCity",
                                  "CountryId": "AttrCityCountryId"}),
            on="AttractionCityId", how="left"
        )

    # ── Resolve VisitMode: join Mode table if VisitMode is stored as numeric IDs ──
    if not mode_df.empty:
        mode_id_col   = mode_df.columns[0]   # e.g. VisitModeId
        mode_name_col = mode_df.columns[1]   # e.g. VisitMode
        vm_series = df["VisitMode"].astype(str)
        numeric_frac = vm_series.str.isnumeric().mean()
        if numeric_frac > 0.5:
            print("  VisitMode detected as numeric IDs — resolving via Mode table …")
            df = df.rename(columns={"VisitMode": "_VisitModeId"})
            df["_VisitModeId"] = df["_VisitModeId"].astype(str)
            mode_lookup = mode_df.copy()
            mode_lookup[mode_id_col] = mode_lookup[mode_id_col].astype(str)
            mode_lookup = mode_lookup.rename(columns={
                mode_id_col:   "_VisitModeId",
                mode_name_col: "VisitMode"
            })[["_VisitModeId", "VisitMode"]]
            df = df.merge(mode_lookup, on="_VisitModeId", how="left")
            df["VisitMode"] = df["VisitMode"].fillna(df["_VisitModeId"])
            df.drop(columns=["_VisitModeId"], inplace=True)
            print(f"  VisitMode values after resolution: {df['VisitMode'].value_counts().to_dict()}")

    print(f"Master dataframe shape: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────
# 4.  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────

CAT_COLS = ["Continent", "Region", "Country", "AttractionType", "Season"]


def feature_engineering(df: pd.DataFrame):
    """Add season, aggregates, encodings. Returns (df, le_mode, le_dict)."""

    def get_season(m):
        if m in [12, 1, 2]:  return "Winter"
        elif m in [3, 4, 5]: return "Spring"
        elif m in [6, 7, 8]: return "Summer"
        else:                 return "Autumn"

    df["Season"] = df["VisitMonth"].apply(get_season)

    # User-level aggregates
    user_avg = df.groupby("UserId")["Rating"].mean().rename("UserAvgRating")
    df = df.merge(user_avg, on="UserId", how="left")

    # Attraction-level aggregates
    attr_avg   = df.groupby("AttractionId")["Rating"].mean().rename("AttrAvgRating")
    attr_count = df.groupby("AttractionId")["UserId"].count().rename("AttrVisitCount")
    df = df.merge(attr_avg,   on="AttractionId", how="left")
    df = df.merge(attr_count, on="AttractionId", how="left")

    # Encode VisitMode label (classification target)
    le_mode = LabelEncoder()
    df["VisitModeLabel"] = le_mode.fit_transform(df["VisitMode"].astype(str))
    print("VisitMode classes:", dict(enumerate(le_mode.classes_)))

    # Encode categorical columns
    le_dict = {}
    for col in CAT_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str).fillna("Unknown"))
            le_dict[col] = le

    print(f"Shape after feature engineering: {df.shape}")
    return df, le_mode, le_dict


# ─────────────────────────────────────────────────────────────────
# 5.  CONVENIENCE WRAPPER
# ─────────────────────────────────────────────────────────────────

def run_pipeline(base_path: str):
    """End-to-end: load → clean → merge → features."""
    ds           = load_datasets(base_path)
    ds           = clean_datasets(ds)
    df           = build_master(ds)
    df, le_mode, le_dict = feature_engineering(df)
    return df, ds["item"], ds["attr_type"], le_mode, le_dict
