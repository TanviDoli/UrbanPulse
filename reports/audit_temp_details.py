import pandas as pd

df = pd.read_csv('data/raw/chennai_daily_temperature_1951_2024.csv')
print("Total rows:", len(df))
print("Exact duplicates:", df.duplicated().sum())

# Date info
dates = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
print("Date min:", dates.min(), "Date max:", dates.max())
print("Null dates:", dates.isnull().sum())

# Duplicates in Date
print("Duplicate dates:", dates.duplicated().sum())

# Check missing dates in the daily sequence
full_daily = pd.date_range('1951-01-01', '2024-06-22', freq='D')
missing_dates = full_daily.difference(dates)
print("Missing calendar dates in sequence:", len(missing_dates))

# Inspect non-numeric entries in Rain, Temp Max, Temp Min
for col in ['Rain', 'Temp Max', 'Temp Min']:
    num = pd.to_numeric(df[col], errors='coerce')
    bad = df[col][num.isnull() & df[col].notnull()]
    print(f"\nNon-numeric in {col} ({len(bad)} rows):")
    print(bad.value_counts().head(10))
