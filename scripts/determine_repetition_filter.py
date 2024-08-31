import yaml
import numpy as np

original_filters = {
    "dup_line_frac": 0.3,
    "dup_para_frac": 0.3,
    "dup_line_char_frac": 0.2,
    "dup_para_char_frac": 0.2,
    "top_2_gram": 0.2,
    "top_3_gram":  0.18,
    "top_4_gram": 0.16,
    "duplicated_5_grams": 0.15,
    "duplicated_6_grams": 0.14,
    "duplicated_7_grams":  0.13,
    "duplicated_8_grams":  0.12,
    "duplicated_9_grams":  0.11,
    "duplicated_10_grams": 0.10, 
}

RAW_STATS_PATH = "./wiki_raw"

with open(f"{RAW_STATS_PATH}/en.yml", "r") as f:
    en_raw_stats = yaml.safe_load(f)
    print(en_raw_stats.keys())


print("Mean/std formulas")
for key, original_filter_value in original_filters.items():
    raw_values = en_raw_stats[key]
    mean = np.mean(raw_values)
    std = np.std(raw_values)
    # orig_val = mean + x*std
    # x = (orig_val - mean) / std
    x = (original_filter_value - mean) / std
    print(f"\t{key}: {original_filter_value:0.4f} = {mean:0.4f} + {x:0.4f} * {std:0.4f}")
    print(f"\t\tSanity check: {mean + x * std:0.4f}")

print("Quantile formulas")
for key, original_filter_value in original_filters.items():
    raw_values = en_raw_stats[key]
    q = sum(np.array(raw_values) < original_filter_value) / len(raw_values)
    print(f"\t{key}: {original_filter_value} = q(raw_values, {q:0.4f})")
    print(f"\t\tSanity check: {np.quantile(raw_values, q):0.4f}")