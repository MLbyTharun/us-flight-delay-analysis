import pandas as pd

INPUT_FILE = "flights_sample_3m.csv.zip"
OUTPUT_FILE = "flights_2022_sample_1_5m.csv"

CHUNKSIZE = 500_000
TARGET_ROWS = 1_500_000

chunks = []
rows_collected = 0

for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNKSIZE):
    remaining = TARGET_ROWS - rows_collected
    if remaining <= 0:
        break

    if len(chunk) > remaining:
        chunk = chunk.iloc[:remaining]

    chunks.append(chunk)
    rows_collected += len(chunk)

    print(f"Collected {rows_collected} rows")

df_final = pd.concat(chunks, ignore_index=True)
df_final.to_csv(OUTPUT_FILE, index=False)

print("✅ DONE")
print("Final shape:", df_final.shape)
print("Saved as:", OUTPUT_FILE)
