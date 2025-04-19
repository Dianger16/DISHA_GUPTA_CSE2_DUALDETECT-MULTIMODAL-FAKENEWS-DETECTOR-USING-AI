import pandas as pd

fake_file = 'gossipcop_fake.csv'
real_file = 'gossipcop_real.csv'

# Check fake dataset
try:
    fake_df = pd.read_csv(fake_file)
    print(f"Columns in {fake_file}: {fake_df.columns.tolist()}")
except FileNotFoundError:
    print(f"{fake_file} not found.")

# Check real dataset
try:
    real_df = pd.read_csv(real_file)
    print(f"Columns in {real_file}: {real_df.columns.tolist()}")
except FileNotFoundError:
    print(f"{real_file} not found.")

import pandas as pd

fake_file = 'gossipcop_fake.csv'
real_file = 'gossipcop_real.csv'

# Check fake dataset
try:
    fake_df = pd.read_csv(fake_file)
    print(f"Columns in {fake_file}: {fake_df.columns.tolist()}")
except FileNotFoundError:
    print(f"{fake_file} not found.")

# Check real dataset
try:
    real_df = pd.read_csv(real_file)
    print(f"Columns in {real_file}: {real_df.columns.tolist()}")
except FileNotFoundError:
    print(f"{real_file} not found.")
