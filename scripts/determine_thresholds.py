import yaml
import numpy as np

original_filters = {
    "min_avg_word_length": 3,
    "max_avg_word_length": 10,
    "alpha_ratio": 0.8,
    "line_punct_ratio": 0.12,
    "short_line_ratio": 0.67,
    "new_line_ratio": 0.3,
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


# From https://gist.github.com/robbibt/c7ec5f0cb3e4e0cee5ed3156bcb666de
def weighted_median(values, weights):
    values = np.array(values)
    weights = np.array(weights)
    sort_indices = np.argsort(values)
    values_sorted = values[sort_indices]
    weights_sorted = weights[sort_indices]  
    cumsum = weights_sorted.cumsum()
    cutoff = weights_sorted.sum() / 2.
    return values_sorted[cumsum >= cutoff][0]

with open(f"{RAW_STATS_PATH}/en.yml", "r") as f:
    en_raw_stats = yaml.safe_load(f)
    print(en_raw_stats.keys())


print("Mean/std formulas")
for key, original_filter_value in original_filters.items():
    if key in ["max_avg_word_length", "min_avg_word_length"]:
        length_counter = en_raw_stats["length_counter"]
        lengths = list(length_counter.keys())
        freqs = list(length_counter.values())
        mean = np.average(lengths, weights=freqs)
        std =  np.sqrt(np.cov(lengths, fweights=freqs))
    else:
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
    if key in ["max_avg_word_length", "min_avg_word_length"]:
        continue
    raw_values = en_raw_stats[key]
    q = sum(np.array(raw_values) < original_filter_value) / len(raw_values)
    print(f"\t{key}: {original_filter_value} = q(raw_values, {q:0.4f})")
    print(f"\t\tSanity check: {np.quantile(raw_values, q):0.4f}")


print("Median/std formulas")
for key, original_filter_value in original_filters.items():
    if key in ["max_avg_word_length", "min_avg_word_length"]:
        length_counter = en_raw_stats["length_counter"]
        lengths = list(length_counter.keys())
        freqs = list(length_counter.values())
        median = weighted_median(lengths, freqs)
        std =  np.sqrt(np.cov(lengths, fweights=freqs))
    else:
        raw_values = en_raw_stats[key]
        median = np.median(raw_values)
        std = np.std(raw_values)
    # orig_val = median + x*std
    # x = (orig_val - median) / std
    x = (original_filter_value - median) / std
    print(f"\t{key}: {original_filter_value:0.4f} = {median:0.4f} + {x:0.4f} * {std:0.4f}")
    print(f"\t\tSanity check: {median + x * std:0.4f}")