import json
import os

base_path = "/iopsstor/scratch/cscs/bmessmer/data/datatrove/multi_lingual_50000_stopwords_nltk/base_processing"
languages = ["ru", "de", "es", "fr", "it", "pt", "nl", "pl"]
DUMP = "CC-MAIN-2023-50"

# Iterate through each language and process the JSON file
print(f"================> updated {base_path}")
for lang in languages:
    file_path = os.path.join(base_path, lang, "logs", "base_processing", DUMP, "stats.json")
    
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            
            # Accessing the third item in the list under the top level
            if len(data) > 2:  # Ensure there's at least 3 items
                stats_data = data[2].get("stats", {})
                total_num_docs = data[0].get("stats", {}).get("documents", {}).get("total", {})
                
                # Extract the required fields from the "stats" dictionary
                dropped_gopher_above = stats_data.get("dropped_gopher_above_avg_threshold", {}).get("total", 0)
                dropped_gopher_below = stats_data.get("dropped_gopher_below_avg_threshold", {}).get("total", 0)
                stop_words_gopher = stats_data.get("dropped_gopher_enough_stop_words", {}).get("total", 0)
                stats_total = stats_data.get("total", {}).get("total", 0)
                dropped_total = stats_data.get("dropped", {}).get("total", 0)
                
                # Calculating ratios
                dropped_gopher_above_ratio = dropped_gopher_above / dropped_total if dropped_total else 0
                dropped_gopher_below_ratio = dropped_gopher_below / dropped_total if dropped_total else 0
                stop_words_gopher_ratio = stop_words_gopher / dropped_total if dropped_total else 0
                dropped_total_ratio = dropped_total / stats_total if stats_total else 0

                filter_stats_data = data[3].get("stats", {})
                list_filter = filter_stats_data.get("dropped_Suspected list", {}).get("total", 0)
                filter_processed_total = filter_stats_data.get("total", {}).get("total", 0)
                list_filter_ratio = list_filter / filter_processed_total if filter_processed_total else 0
                
                print(f"Language: {lang}")
                print(f"Dropped Gopher Above Avg Threshold: {dropped_gopher_above} ({dropped_gopher_above_ratio:.2%} of dropped total)")
                print(f"Dropped Gopher Below Avg Threshold: {dropped_gopher_below} ({dropped_gopher_below_ratio:.2%} of dropped total)")
                print(f"Dropped Stopwords Threshold: {stop_words_gopher} ({stop_words_gopher_ratio:.2%} of dropped total)")
                print(f"Dropped List Threshold: {list_filter} ({list_filter_ratio:.2%} of dropped total)")
                print(f"Total Processed: {total_num_docs}")
                print(f"Total Processed (Quality filter): {stats_total}")
                print(f"Dropped Total: {dropped_total} ({dropped_total_ratio:.2%} of stats total)")
                print("----------")
            else:
                print(f"Data for language {lang} does not contain enough items to process.")
                
    except FileNotFoundError:
        print(f"File not found for language {lang}: {file_path}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file for language {lang}: {file_path}")