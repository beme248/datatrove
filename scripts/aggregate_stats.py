import json
import os
import csv
from datatrove.utils.typeshelper import Languages


DOC_LIMIT = 5000
RUN_SUFFIX = "top50_stopwords_p_thresh_0_008"
RUN_NAME = f"multi_lingual_{DOC_LIMIT}_{RUN_SUFFIX}"

# Define your CSV file name
csv_file_name = f'language_statistics_{RUN_NAME}.csv'

# Define the CSV column names
fieldnames = [ "Language", "Dropped Gopher Above Avg Threshold Ratio",
              "Dropped Gopher Above Avg Threshold Count", "Dropped Gopher Below Avg Threshold Ratio",
              "Dropped Gopher Below Avg Threshold Count", "Dropped Stopwords Threshold Ratio",
              "Dropped Stopwords Threshold Count", "Dropped Alpha Ratio", "Dropped Alpha Count", "Dropped List Threshold Ratio", "Dropped List Threshold Count",
              "Total Processed", "Total Processed (Quality filter)", "Dropped Total Ratio", "Dropped Total Count",
              "Total Retained Count", "Total Retained Ratio" ]

base_path = f"/iopsstor/scratch/cscs/bmessmer/data/datatrove/{RUN_NAME}/base_processing"
languages = [ Languages.english, Languages.spanish, Languages.portuguese, Languages.italian, Languages.french, Languages.romanian, Languages.german, Languages.latin, Languages.czech, Languages.danish, Languages.finnish, Languages.greek, Languages.norwegian, Languages.polish, Languages.russian, Languages.slovenian, Languages.swedish, Languages.turkish, Languages.dutch, Languages.chinese, Languages.japanese, Languages.vietnamese, Languages.indonesian, Languages.persian, Languages.korean, Languages.arabic, Languages.thai, Languages.hindi, Languages.bengali, Languages.tamil, Languages.hungarian, Languages.ukrainian, Languages.slovak, Languages.bulgarian, Languages.catalan, Languages.croatian, Languages.serbian, Languages.lithuanian, Languages.estonian, Languages.hebrew, Languages.latvian, Languages.serbocroatian, Languages.albanian, Languages.azerbaijani, Languages.icelandic, Languages.macedonian, Languages.georgian, Languages.galician, Languages.armenian, Languages.basque ]

DUMP = "CC-MAIN-2023-50"

# Iterate through each language and process the JSON file
print(f"================> updated {base_path}")

# Writing to the CSV file
with open(csv_file_name, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    # Write the header
    writer.writeheader()

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
                    alpha_gopher = stats_data.get("dropped_gopher_below_alpha_threshold", {}).get("total", 0)
                    stats_total = stats_data.get("total", {}).get("total", 0)
                    dropped_total = stats_data.get("dropped", {}).get("total", 0)
                    
                    # Calculating ratios
                    dropped_gopher_above_ratio = dropped_gopher_above / dropped_total if dropped_total else 0
                    dropped_gopher_below_ratio = dropped_gopher_below / dropped_total if dropped_total else 0
                    stop_words_gopher_ratio = stop_words_gopher / dropped_total if dropped_total else 0
                    alpha_ratio = alpha_gopher / dropped_total if dropped_total else 0
                    dropped_total_ratio = dropped_total / stats_total if stats_total else 0

                    filter_stats_data = data[3].get("stats", {})
                    list_filter = filter_stats_data.get("dropped_Suspected list", {}).get("total", 0)
                    filter_processed_total = filter_stats_data.get("total", {}).get("total", 0)
                    list_filter_ratio = list_filter / filter_processed_total if filter_processed_total else 0

                    clean_total = data[-1].get("stats", {}).get("total", {}).get("total", {})
                    clean_total_ratio = clean_total / total_num_docs if clean_total else 0

                    row = {
                        "Language" : lang,
                        "Dropped Gopher Above Avg Threshold Ratio": dropped_gopher_above_ratio,
                        "Dropped Gopher Above Avg Threshold Count": dropped_gopher_above,
                        "Dropped Gopher Below Avg Threshold Ratio": dropped_gopher_below_ratio,
                        "Dropped Gopher Below Avg Threshold Count": dropped_gopher_below,
                        "Dropped Stopwords Threshold Ratio": stop_words_gopher_ratio,
                        "Dropped Stopwords Threshold Count": stop_words_gopher,
                        "Dropped Alpha Ratio": alpha_ratio,
                        "Dropped Alpha Count": alpha_gopher,
                        "Dropped List Threshold Ratio": list_filter_ratio,
                        "Dropped List Threshold Count": list_filter,
                        "Total Processed": total_num_docs,
                        "Total Processed (Quality filter)": stats_total,
                        "Dropped Total Ratio": dropped_total_ratio,
                        "Dropped Total Count": dropped_total,
                        "Total Retained Count": clean_total,
                        "Total Retained Ratio": clean_total_ratio,
                    } 

                    writer.writerow(row)
                else:
                    print(f"Data for language {lang} does not contain enough items to process.")
                    
        except FileNotFoundError:
            print(f"File not found for language {lang}: {file_path}")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file for language {lang}: {file_path}")