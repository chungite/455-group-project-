import pandas as pd

customers = pd.read_csv('csv_mania/customers.csv')

# Encode membership status
customers = pd.get_dummies(
    customers,
    columns=['club_member_status'],
    dtype=int
)

# Encode fashion news frequency
customers = pd.get_dummies(
    customers,
    columns=['fashion_news_frequency'],
    dtype=int
)
customers.drop(columns=['customer_id','FN', 'Active','postal_code'], inplace=True)
# Save the encoded dataset
customers.to_csv('customers_encoded_one_hot.csv', index=False)
