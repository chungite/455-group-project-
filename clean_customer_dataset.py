import pandas as pd

# 1. Load the CSV file
df = pd.read_csv('csv_mania/customers.csv')

# 2. Drop the column by name
# Use inplace=True to modify the existing DataFrame directly
df.drop(columns=['FN', 'Active','postal_code'], inplace=True)

df.to_csv('cleaned_customer.csv', index=False)
