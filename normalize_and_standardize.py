import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


df = pd.read_csv(
    'csv_mania/articles.csv',
    na_values=['n/a', '--', 'Unknown']
)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

columns_to_check = [
    'graphical_appearance_no',
    'colour_group_code',
    'perceived_colour_value_id',
    'perceived_colour_master_id'
]

# Remove rows where any specified column has a value of -1
rows_with_missing_values = df[df.isna().any(axis=1)].copy()
rows_with_missing_values.to_csv('missing_report.csv', index=False)

df = df[~(df[columns_to_check] == -1).any(axis=1)].copy()
# Map description keywords to garment groups
keyword_mapping = {
    'denim': {
        'garment_group_name': 'Trousers Denim',
        'product_type_name': 'Trousers', 
        'product_type_no' : 272,
        'product_group_name' :'Garment Lower body',
    },
        'jeans': {
        'garment_group_name': 'Trousers Denim',
        'product_type_name': 'Trousers', 
        'product_type_no' : 272,
        'product_group_name' :'Garment Lower body',
    },
        'trousers': {
        'garment_group_name': 'Trousers',
        'product_type_name': 'Trousers', 
        'product_type_no' : 272,
        'product_group_name' :'Garment Lower body',
    },
            'pants': {
        'garment_group_name': 'Trousers',
        'product_type_name': 'Trousers', 
        'product_type_no' : 272,
        'product_group_name' :'Garment Lower body',
    },
            'joggers': {
        'garment_group_name': 'Trousers',
        'product_type_name': 'Trousers', 
        'product_type_no' : 272,
        'product_group_name' :'Garment Lower body',
    },
            'jacket': {
        'garment_group_name': 'Outdoor',
        'product_type_name': 'Jacket', 
        'product_type_no' : 262,
        'product_group_name' :'Garment Upper body',
    },
                'coat': {
        'garment_group_name': 'Outdoor',
        'product_type_name': 'Jacket', 
        'product_type_no' : 262,
        'product_group_name' :'Garment Upper body',
    },
            'parka': {
        'garment_group_name': 'Outdoor',
        'product_type_name': 'Jacket', 
        'product_type_no' : 262,
        'product_group_name' :'Garment Upper body',
    },
            'skirts': {
        'garment_group_name': 'Skirts',
        'product_type_name': 'Skirt', 
        'product_type_no' : 275,
        'product_group_name' :'Garment Lower body',
    },

     
     #          'breifs': {
     #   'garment_group_name': 'Under-, Nightwear',
     #   'product_type_name': 'Shorts', 
     #   'product_type_no' : 274,
     #   'product_group_name' :'Garment Lower body',
    #},
               'shorts': {
        'garment_group_name': 'Shorts',
        'product_type_name': 'Shorts', 
        'product_type_no' : 274,
        'product_group_name' :'Garment Lower body',
    },    
               'cardigan': {
        'garment_group_name': 'Knitwear',
        'product_type_name': 'Sweater', 
        'product_type_no' : 252 ,
        'product_group_name' :'Garment Upp body',
    },
               'sweater': {
        'garment_group_name': 'Knitwear',
        'product_type_name': 'Sweater', 
        'product_type_no' : 252 ,
        'product_group_name' :'Garment Upper body',
    },
               'jumper': {
        'garment_group_name': 'Knitwear',
        'product_type_name': 'Sweater', 
        'product_type_no' : 252 ,
        'product_group_name' :'Garment Upper body',
    },
               'knit': {
        'garment_group_name': 'Knitwear',
        'product_type_name': 'Sweater', 
        'product_type_no' : 252 ,
        'product_group_name' :'Garment Upper body',
    },
               'blouse': {
        'garment_group_name': 'Blouse',
        'product_type_name': 'Blouse', 
        'product_type_no' : 258 ,
        'product_group_name' :'Garment Upper body',
    },
                   'T-shirt': {
        'garment_group_name': 'Jersey Basic',
        'product_type_name': 'T-shirt', 
        'product_type_no' : 255 ,
        'product_group_name' :'Garment Upper body',
    },
                   'shirt': {
        'garment_group_name': 'Jersey Basic',
        'product_type_name': 'Shirt', 
        'product_type_no' : 255 ,
        'product_group_name' :'Garment Upper body',
    },
                 'shirt': {
        'garment_group_name': 'Jersey Basic',
        'product_type_name': 'Shirt', 
        'product_type_no' : 259 ,
        'product_group_name' :'Garment Upper body',
    },

                 'jersey': {
        'garment_group_name': 'Jersey Basic',
        'product_type_name': 'Shirt', 
        'product_type_no' : 259 ,
        'product_group_name' :'Garment Upper body',
    },
                 'shirt': {
        'garment_group_name': 'Jersey Basic',
        'product_type_name': 'Shirt', 
        'product_type_no' : 259 ,
        'product_group_name' :'Garment Upper body',
    },
    
                 'sweatshirt': {
        'garment_group_name': 'Jersey Fancy',
        'product_type_name': 'Shirt', 
        'product_type_no' : 259 ,
        'product_group_name' :'Garment Upper body',
    },
                 'v-neck': {
        'garment_group_name': 'Jersey Fancy',
        'product_type_name': 'Top', 
        'product_type_no' : 254 ,
        'product_group_name' :'Garment Upper body',
    },

                     'top': {
        'garment_group_name': 'Jersey Basic',
        'product_type_name': 'Top', 
        'product_type_no' : 254 ,
        'product_group_name' :'Garment Upper body',
    },
                     'dress': {
        'garment_group_name': 'Dresses Ladies',
        'product_type_name': 'Dress', 
        'product_type_no' : 265 ,
        'product_group_name' :'Garment Full body',
    },
                      'bra': {
        'garment_group_name': 'Under-, Nightwear',
        'product_type_name': 'Bra', 
        'product_type_no' : 306 ,
        'product_group_name' :'Garment Upper body',
    },   
                      'briefs': {
        'garment_group_name': 'Under-, Nightwear',
        'product_type_name': 'Underwear bottom', 
        'product_type_no' : 286 ,
        'product_group_name' :'Garment Lower body',
    },     
                      'hipster': {
        'garment_group_name': 'Under-, Nightwear',
        'product_type_name': 'Underwear bottom', 
        'product_type_no' : 286 ,
        'product_group_name' :'Garment Lower body',
    },       
                'tights': {
        'garment_group_name': 'Socks and Tights',
        'product_type_name': 'Trousers', 
        'product_type_no' : 272,
        'product_group_name' :'Garment Lower body',
    },
                'socks': {
        'garment_group_name': 'Socks and Tights',
        'product_type_name': 'Trousers', 
        'product_type_no' : 272,
        'product_group_name' :'Garment Lower body',
    },
                'shoes': {
        'garment_group_name': 'Shoes',
        'product_type_name': 'Flat Shoe', 
        'product_type_no' : 144,
        'product_group_name' :'Shoes',
    },    
                'sandals': {
        'garment_group_name': 'Shoes',
        'product_type_name': 'Flat Shoe', 
        'product_type_no' : 144,
        'product_group_name' :'Shoes',
    },   
                    'hat': {
        'garment_group_name': 'Accessories',
        'product_type_name': 'Hat/beanie', 
        'product_type_no' : 75,
        'product_group_name' :'Accessories',
    },      

                         'patch': {
        'garment_group_name': 'Accessories',
        'product_type_name': ' Other accessories', 
        'product_type_no' : 78,
        'product_group_name' :'Accessories',
    },           
  
                         'band': {
        'garment_group_name': 'Accessories',
        'product_type_name': ' Other accessories', 
        'product_type_no' : 78,
        'product_group_name' :'Accessories',
    }, 
                         'case': {
        'garment_group_name': 'Small Accessories',
        'product_type_name': ' Other accessories', 
        'product_type_no' : 78,
        'product_group_name' :'Accessories',
    }, 
                         'longjohns': {
        'garment_group_name': 'Accessories',
        'product_type_name': ' Other accessories', 
        'product_type_no' : 78,
        'product_group_name' :'Accessories',
    }, 
                         'bag': {
        'garment_group_name': 'Accessories',
        'product_type_name': ' Other accessories', 
        'product_type_no' : 78,
        'product_group_name' :'Accessories',
    }, 
}

def infer_garment_group(row):
    columns_to_check = ['product_type_no']

    if pd.isna(row['detail_desc']):
        return row

    desc = str(row['detail_desc']).lower()

    for keyword, replacements in keyword_mapping.items():
        if keyword.lower() in desc:
            for column, value in replacements.items():
                if (
                    pd.isna(row[column])
                    or (
                        column in columns_to_check
                        and row[column] == -1
                    )
                ):
                    row[column] = value

            break

    return row


# Record missing values before fixing
missing_before = df['garment_group_name'].isna()

# Fill only missing garment_group_name values
df = df.apply(infer_garment_group, axis=1)
df_cleaned = df.dropna(subset=['detail_desc'])
df_cleaned = df_cleaned.dropna(subset=['garment_group_name'])
# Save only rows where garment_group_name was fixed
fixed_rows = df[
    missing_before & df['garment_group_name'].notna()
].copy()


# Save the full dataset with only garment_group_name changed
df_cleaned.to_csv('cleaned_article.csv', index=False)


customers = pd.read_csv('csv_mania/customers.csv')
customers.drop_duplicates(inplace=True)
# 2. Drop the column by name
# Use inplace=True to modify the existing DataFrame directly
customers.drop(columns=['FN', 'Active','postal_code'], inplace=True)



customers["club_member_status"] = (
    customers["club_member_status"].fillna("Missing")
)

customers["fashion_news_frequency"] = (
    customers["fashion_news_frequency"].fillna("Missing")
)

# Impute missing ages with the median
customers["age"] = customers["age"].fillna(
    customers["age"].median()
)

# Verify that missing values are handled
print(customers[["age", "club_member_status",
                 "fashion_news_frequency"]].isna().sum())



customers.to_csv("cleaned_customer.csv", index=False)

missing_rows = customers[customers.isna().any(axis=1)]

print(missing_rows)
print("Number of rows with missing values:", len(missing_rows))
print((customers.isna().mean() * 100).round(2))



df = pd.read_csv('csv_mania/transactions_train.csv')
# Convert column to datetime
# Convert the t_dat column to datetime, keeping the DataFrame
df['t_dat'] = pd.to_datetime(df['t_dat'], errors='coerce')

# Find rows containing missing values
missing_rows = df[df.isna().any(axis=1)]

print(missing_rows)
print("Number of rows with missing values:", len(missing_rows))

# Show missing-value counts and percentages for each column
print("\nMissing values per column:")
print(df.isna().sum())

print("\nMissing-value percentages:")
print((df.isna().mean() * 100).round(2))

# Save the cleaned dataset
df.to_csv('cleaned_transactions_training.csv', index=False)





# 1. Load article and customer information
articles = pd.read_csv("cleaned_article.csv")
customers = pd.read_csv("cleaned_customer.csv")

# Keep the customer columns needed for feature engineering
customer_columns = [
    "customer_id",
    "age",
    "club_member_status",
    "fashion_news_frequency"
]
customers = customers[customer_columns].copy()

# 2. Read transaction columns
transactions = pd.read_csv(
    "cleaned_transactions_training.csv",
    usecols=[
        "article_id",
        "customer_id",
        "price",
        "sales_channel_id"
    ],
    dtype={
        "article_id": "int32",
        "customer_id": "category",
        "sales_channel_id": "int8"
    }
)

# Make customer ID types consistent before joining
transactions["customer_id"] = transactions["customer_id"].astype(str)
customers["customer_id"] = customers["customer_id"].astype(str)

# 3. Aggregate transaction features by article
transaction_features = transactions.groupby(
    "article_id",
    observed=True
).agg(
    purchase_count=("article_id", "size"),
    unique_customers=("customer_id", "nunique"),
    average_price=("price", "mean"),
    channel_1_share=("sales_channel_id", lambda s: (s == 1).mean()),
    channel_2_share=("sales_channel_id", lambda s: (s == 2).mean())
).reset_index()

# 4. Join customers to transactions using customer_id
customer_transactions = transactions[
    ["article_id", "customer_id"]
].drop_duplicates()

customer_transactions = customer_transactions.merge(
    customers,
    on="customer_id",
    how="left"
)

# 5. Create customer features for each article
# Each customer counts once per article.
customer_features = customer_transactions.groupby(
    "article_id",
    observed=True
).agg(
    average_buyer_age=("age", "mean")
).reset_index()

# Calculate the proportion of buyers in each customer category
membership_shares = pd.crosstab(
    customer_transactions["article_id"],
    customer_transactions["club_member_status"],
    normalize="index"
)
membership_shares.columns = [
    f"club_member_share_{col}" for col in membership_shares.columns
]
membership_shares = membership_shares.reset_index()

news_shares = pd.crosstab(
    customer_transactions["article_id"],
    customer_transactions["fashion_news_frequency"],
    normalize="index"
)
news_shares.columns = [
    f"fashion_news_share_{col}" for col in news_shares.columns
]
news_shares = news_shares.reset_index()

customer_features = customer_features.merge(
    membership_shares, on="article_id", how="left"
).merge(
    news_shares, on="article_id", how="left"
)

del transactions, customer_transactions, customers

# 6. Combine article, transaction, and customer features
df = articles.merge(
    transaction_features,
    on="article_id",
    how="left"
).merge(
    customer_features,
    on="article_id",
    how="left"
)

# Fill transaction features for articles never purchased
zero_features = [
    "unique_customers",
    "average_price",
    "channel_1_share",
    "channel_2_share"
]
df[zero_features] = df[zero_features].fillna(0)
df["never_purchased"] = df["purchase_count"].isna().astype(int)
df[["purchase_count", "unique_customers"]] = df[["purchase_count", "unique_customers"]].fillna(0)
df["channel_1_share"] = df["channel_1_share"].fillna(0)
df = df.drop(columns=["channel_2_share"], errors="ignore")
# Peform Normalization and Standardization for the aggregated features
#numeric_features = [
#    "average_price",
#    "average_buyer_age",
#    "channel_1_share"
#]

#Channel 1 and 2 are normalized into proportion purchased through in store (1) or online(2)
# Club member share  and fashion news share are also normalized in proportion of what percentage of purchasers is
# regular, monthly, etc
# Standardize using the Z score for the following features
numeric_features = [
    "purchase_count",
    "unique_customers",
    "average_price",
    "average_buyer_age"
]
count_features = ["purchase_count", "unique_customers"]


#df_scaled = df.copy()

# Fill missing values before scaling


#for col in numeric_features:
#    df_scaled[col] = df_scaled[col].fillna(
#        df_scaled[col].median()
#    )

#scaler = StandardScaler()
#df_scaled[count_features] = np.log1p(df_scaled[count_features])
#
#df_scaled[numeric_features] = scaler.fit_transform(
#    df_scaled[numeric_features]
#)
# Fill transaction features for articles never purchased
#print(df_scaled[numeric_features].describe())




##### one hot encoding ######



share_cols = [c for c in df.columns
              if c.startswith(("channel_1_share", "club_member_share_", "fashion_news_share_"))]
df[share_cols] = df[share_cols].fillna(0)
df = df.drop(columns=["channel_2_share"], errors="ignore")   # redundant with channel_1_share

# average_price / average_buyer_age stay NaN for unpurchased articles -> imputed in the pipeline
# (make sure you remove average_price from any fillna(0) step)
#save initial purchase count for mutual info 
df["purchase_count_raw"] = df["purchase_count"]
# --- Outliers ---
for col in ["average_price", "average_buyer_age"]:
    lo, hi = df[col].quantile([0.01, 0.99])
    df[col] = df[col].clip(lo, hi)

for col in ["purchase_count", "unique_customers"]:
    df[col] = np.log1p(df[col])
    df[col] = df[col].clip(upper=df[col].quantile(0.99))

# --- Features ---
numeric_features = ["purchase_count", "unique_customers", "average_price", "average_buyer_age"]
passthrough_features = share_cols + ["never_purchased"]   # already in [0, 1]

# 8. Choose categorical article features
categorical_features = [
    "product_type_name",
    "product_group_name",
    "colour_group_name",
    "perceived_colour_value_name",
    "perceived_colour_master_name",
    "department_name",
    "index_name",
    "index_group_name",
    "section_name",
    "garment_group_name",
    "graphical_appearance_name"
]


X = df.set_index("article_id")[categorical_features + numeric_features + passthrough_features]

preprocessor = ColumnTransformer([
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ]), categorical_features),
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric_features),
    ("pass", "passthrough", passthrough_features),
])

X_encoded = preprocessor.fit_transform(X)

preview = pd.DataFrame(
    X_encoded[:10].toarray(),
    columns=preprocessor.get_feature_names_out(),
    index=X.index[:10],
)
preview.to_csv("test.csv")

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),  # dense output
        ]), categorical_features),
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_features),
        ("pass", "passthrough", passthrough_features),
    ],
    remainder="drop",
    sparse_threshold=0,              # force a dense result
    verbose_feature_names_out=False, # cleaner column names (no "cat__" prefixes)
)

# Return a pandas DataFrame (keeps the article_id index and column names)
preprocessor.set_output(transform="pandas")

X_encoded = preprocessor.fit_transform(X)

print("Original feature columns:", X.shape[1])
print("Encoded feature columns:", X_encoded.shape[1])

# Optional: shrink file/memory size (one-hot columns are 0/1 anyway)
X_encoded = X_encoded.astype("float32")

X_encoded.to_csv("encoded_features.csv", index_label="article_id")

######## Mutual information for feature selection #########

# 7. Define popularity category


df["purchase_category"] = pd.cut(
    df["purchase_count_raw"],
    bins=[-1, 0, 24, 173, float("inf")],
    labels=["Never purchased", "Low", "Medium", "High"]
)
print(df["purchase_category"].value_counts())   # sanity check: all 4 classes should appear


# 8. Choose categorical article features


categorical_features = [
    c for c in categorical_features if c in df.columns
]

# Customer categories are represented by proportions,
# so they can be treated as numerical features.

# Remove duplicate feature names and keep available columns

all_numeric_features = ["average_price", "average_buyer_age"] + passthrough_features

# 9. Encode article categories
X_cat = df[categorical_features].fillna("Missing").astype(str)

encoder = OrdinalEncoder()
X_cat_encoded = pd.DataFrame(
    encoder.fit_transform(X_cat),
    columns=categorical_features,
    index=df.index
)

X_num = df[all_numeric_features].copy()

# Median-impute missing numerical values
X_num = X_num.fillna(X_num.median()).fillna(0)

X = pd.concat([X_cat_encoded, X_num], axis=1)
y = df["purchase_category"]

valid = y.notna()
X = X.loc[valid]
y = y.loc[valid]

# Encoded article categories are discrete.
# Numerical features, including category proportions, are continuous
# for this MI calculation.
mi_numeric = X_num.columns.tolist()

discrete_mask = (
    [True] * len(categorical_features)
    + [False] * len(mi_numeric)
)

assert len(discrete_mask) == X.shape[1]
# 10. Calculate mutual information
mi_scores = mutual_info_classif(
    X,
    y,
    discrete_features=discrete_mask,
    random_state=42
)

scores = pd.Series(
    mi_scores,
    index=X.columns
).sort_values(ascending=False)

print("Mutual Information scores:")
print(scores)

print("\nTop 10 features:")
print(scores.head(10))

scores.to_csv(
    "mutual_information_scores.csv",
    header=["mi_score"]
)