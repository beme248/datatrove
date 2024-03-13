import pandas as pd
import os

# Given dictionary
language_dict = {
    'Dutch': 'nl',
    'Polish': 'pl',
    'Czech': 'cs',
    'Swedish': 'sv',
    'Hungarian': 'hu',
    'Greek': 'el',
    'Romanian': 'ro',
    'Danish': 'da',
    'Finnish': 'fi',
    'Ukrainian': 'uk',
    'Slovak': 'sk',
    'Norwegian': 'no',
    'Bulgarian': 'bg',
    'Catalan': 'ca',
    'Croatian': 'hr',
    'Latin': 'la',
    'Serbian': 'sr',
    'Slovenian': 'sl',
    'Lithuanian': 'lt'
}

scratch = os.getenv('SCRATCH')
base_path = os.path.join(scratch, 'fork')
wiki_path = os.path.join(base_path, 'datatrove/scripts/data/wiki_list.csv')
estimated_documents_path = os.path.join(base_path, 'datatrove/scripts/data/estimated_documents.csv')

top_50_path = os.path.join(base_path, 'datatrove/scripts/data/language_selection.csv')

# Load the data from files
wiki_list = pd.read_csv(wiki_path)
estimated_documents = pd.read_csv(estimated_documents_path)
# Convert dictionary to DataFrame
dict_df = pd.DataFrame(list(language_dict.items()), columns=['language', 'iso'])

# Preprocessing
wiki_list.drop_duplicates(subset='iso', inplace=True)
wiki_list.dropna(subset=['iso'], inplace=True)
estimated_documents.drop_duplicates(subset='iso', inplace=True)
estimated_documents.dropna(subset=['iso'], inplace=True)
estimated_documents_sorted = estimated_documents.sort_values(by='doc_counts', ascending=False)

# Selecting top ISO codes
iso_codes_wiki_list = wiki_list['iso'].tolist()
additional_needed = max(0, 51 - len(iso_codes_wiki_list))
top_additional_isos = estimated_documents_sorted[~estimated_documents_sorted['iso'].isin(iso_codes_wiki_list)].head(additional_needed)['iso'].tolist()
final_iso_codes = iso_codes_wiki_list + top_additional_isos

# Create a DataFrame for the top 50 ISO codes
final_df = pd.DataFrame(final_iso_codes, columns=['iso'])

# Merge to get document counts from estimated_documents
final_df = pd.merge(final_df, estimated_documents_sorted[['iso', 'doc_counts']], on='iso', how='left')

# Merge to get initial language names from wiki_list
final_df = pd.merge(final_df, wiki_list[['iso', 'language']], on='iso', how='left')

# Merge to overwrite/add language names from dict_df
final_df = pd.merge(final_df, dict_df, on='iso', how='left', suffixes=('', '_dict'))

# Prioritize language names from dict_df and fill missing values
final_df['language'] = final_df['language_dict'].fillna(final_df['language'])
final_df.drop(columns=['language_dict'], inplace=True)

# Print final ISO codes (demonstration purpose)
print("Final ISO Codes List:")
print(final_df['iso'].tolist())
print(final_df['language'].tolist())

# Write the result to a new CSV file
final_df.to_csv(top_50_path, index=False)

print(f"Final list with up to 50 ISO codes and language names has been written to 'final_iso_languages.csv'.")
