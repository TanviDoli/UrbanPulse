import pandas as pd

df = pd.read_csv('data/raw/chennai_daily_rainfall_1991_2023.csv')
print("Total rows:", len(df))
print("Exact duplicate rows:", df.duplicated().sum())

# Date info
dates = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
print("Date min:", dates.min(), "Date max:", dates.max())

# Unique stations
st_counts = df['Station'].value_counts()
print("\nTop 20 stations by record count:")
print(st_counts.head(20))
print(f"\nTotal unique stations: {len(st_counts)}")

# Check duplicate station-date pairs
dup_st_date = df[df.duplicated(subset=['Station', 'Date'], keep=False)]
print(f"\nDuplicate Station-Date pairs: {len(dup_st_date)}")
