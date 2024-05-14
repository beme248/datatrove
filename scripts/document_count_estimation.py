import boto3
import random
from statistics import mean
import gzip
import pandas as pd

def count_documents(bucket_name, prefix):
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    files = []
    for page in page_iterator:
        for obj in page['Contents'][1:]:
            files.append(obj['Key'])

    return files

def read_random_files(bucket_name, files, sample_size=30):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    sampled_files = random.sample(files, min(len(files), sample_size))

    line_counts = []
    for file_key in sampled_files:
        obj = bucket.Object(file_key)
        response = obj.get()
        with gzip.GzipFile(fileobj=response['Body']) as gzipfile:
            content = gzipfile.read().decode('utf-8')
            lines = content.splitlines()
            line_counts.append(len(lines))

    return line_counts

def estimate_total_lines(bucket_name, prefix):
    files = count_documents(bucket_name, prefix)
    total_files = len(files)
    print(f"Total files in folder: {total_files}")

    if total_files == 0:
        return 0

    line_counts = read_random_files(bucket_name, files)
    avg_lines = mean(line_counts)
    print(f"Average lines per file (from sample): {avg_lines}")

    estimated_total_lines = avg_lines * total_files
    return estimated_total_lines

def process_language(lang):
    bucket_name = 'fineweb-data-processing-us-east-1'
    prefix = f'base_processing/non_english/{lang}/CC-MAIN-2023-23'
    estimated_lines = estimate_total_lines(bucket_name, prefix)
    print(f"Estimated total JSONLines in folder: {estimated_lines}")
    return estimated_lines


LANGUAGES = "af als am an ar arz as ast av az azb ba bar bcl be bg bh bn bo bpy br bs bxr ca cbk ce ceb ckb co cs cv cy da de diq dsb dty dv el eml en eo es et eu fa fi fr frr fy ga gd gl gn gom gu gv he hi hif hr hsb ht hu hy ia id ie ilo io is it ja jbo jv ka kk km kn ko krc ku kv kw ky la lb lez li lmo lo lrc lt lv mai mg mhr min mk ml mn mr mrj ms mt mwl my myv mzn nah nap nds ne new nl nn no oc or os pa pam pfl pl pms pnb ps pt qu rm ro ru rue sa sah sc scn sco sd sh si sk sl so sq sr su sv sw ta te tg th tk tl tr tt tyv ug uk ur uz vec vep vi vls vo wa war wuu xal xmf yi yo yue zh".split(
    " "
)

# Collect data
data = []
for lang in LANGUAGES:
    estimated_lines = process_language(lang)
    data.append((lang, estimated_lines))

# Convert to DataFrame
df = pd.DataFrame(data, columns=['Language', 'EstimatedLines'])

# Sort by EstimatedLines in descending order
df_sorted = df.sort_values(by='EstimatedLines', ascending=False)

# Write to CSV
df_sorted.to_csv('language_estimates.csv', index=False)
print("CSV file has been written.")