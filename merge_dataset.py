import pandas as pd

# 1. Load the CSV files into DataFrames
df1 = pd.read_csv('csv_mania/customers.csv')
df2 = pd.read_csv('csv_mania/transactions_train.csv')
df3 = pd.read_csv('csv_mania/articles.csv')
# 2. Join the files side-by-side on a common column
# Options for 'how': 'inner' (only matching IDs), 'left' (all from file1), 'right', or 'outer' (all data)
merged_cust_and_trans = pd.merge(df1, df2, on='customer_id', how='inner')
merged_trans_and_article = pd.merge(df2, df3, on='article_id', how='inner')
# 3. Export to a new CSV file
merged_cust_and_trans.to_csv('joined_customers_and_transactions.csv', index=False)
merged_trans_and_article.to_csv('joined_trans_and_article.csv', index=False)

