
import pandas as pd

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
    'denim': 'Trousers Denim',
    'hat' : 'Accessories',
    'jeans': 'Trousers',
    'jacket': 'Outdoor',
    'coat': 'Outdoor',
    'parka': 'Outdoor',
    'briefs': 'Under-, Nightwear',
    'bra': 'Under-, Nightwear',
    'hipster': 'Under-, Nightwear',
    'cardigan': 'Knitwear',
    'jumper': 'Knitwear',
    'sweater': 'Knitwear',
    'blouse': 'Blouses',
    'socks': 'Socks and Tights',
    'tights': 'Socks and Tights',
    'dress': 'Dresses Ladies',
    'shoes': 'Shoes',
    'trousers': 'Trousers',
    'pants': 'Trousers',
    'shorts': 'Shorts',
    'skirt': 'Skirts'
}

def infer_garment_group(row):
    if pd.isna(row['garment_group_name']) and pd.notna(row['detail_desc']):
        desc = str(row['detail_desc']).lower()

        for keyword, group in keyword_mapping.items():
            if keyword in desc:
                return group

    return row['garment_group_name']


# Record missing values before fixing
missing_before = df['garment_group_name'].isna()

# Fill only missing garment_group_name values
df['garment_group_name'] = df.apply(infer_garment_group, axis=1)
df_cleaned = df.dropna(subset=['detail_desc'])
df_cleaned = df_cleaned.dropna(subset=['garment_group_name'])
# Save only rows where garment_group_name was fixed
fixed_rows = df[
    missing_before & df['garment_group_name'].notna()
].copy()

fixed_rows.to_csv('fixed_garment_groups.csv', index=False)

# Save the full dataset with only garment_group_name changed
df.to_csv('cleaned_data.csv', index=False)

print(f"Garment groups fixed: {len(fixed_rows)}")
print("Saved fixed rows to fixed_garment_groups.csv")
print("Saved full dataset to cleaned_data.csv")