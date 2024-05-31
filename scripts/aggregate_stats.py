import json
import os
import csv
from datatrove.utils.typeshelper import Languages
import fire

languages = [
    "en",
    "de",
    "hr",
    "pt",
    "cs",
    "zh",
    "fr",
    "ru",
    "tr",
    "ar",
    "th",
    "hi",
    "sw",
    "te",
    "ja"]

def create_path_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main(
        filter_mode: str = 'wiki',
        dataset_mode: str = 'cc',
        dump: str = "CC-MAIN-2023-23"):
    output_path = os.path.join("aggregated_filter_statistics")
    create_path_if_not_exists(output_path)

    with open(os.path.join(output_path, f"{dataset_mode}_{filter_mode}_language_statistics.csv"), mode='w', newline='', encoding='utf-8') as file:
        rows = []
        for lang in languages:
            
            file_path = os.path.join("processing", f"multilingual_{dataset_mode}_with {filter_mode}_filters", "logs", dump, lang, "stats.json")
            print(f"path {file_path}")
            try:
                with open(file_path, 'r') as json_file:
                    data = json.load(json_file)
                    
                    # Accessing the third item in the list under the top level
                    if len(data) > 4:  # Ensure there's at least 3 items
                        row = {
                            "Language": lang
                        }
                        total_num_docs = data[0].get("stats", {}).get("documents", {}).get("total", {})
                        for filter_idx in range(1, 3):
                            filter_stats = data[filter_idx].get("stats", {})

                            filter_total = filter_stats["total"]
                            filter_forwarded = filter_stats["forwarded"]
                            total_dropped = filter_stats["dropped"]

                            for k, v in filter_stats.items():
                                if k.startswith("dropped_"):
                                    row[f'absolute_{k}'] = v
                                    row[f'ratio_{k}'] = v / total_dropped

                        row['absolute_dropped'] = total_dropped
                        row['ratio_dropped'] = total_dropped / filter_total
                        row['absolute_forwarded'] =  filter_forwarded
                        row['ratio_forwarded'] =  filter_forwarded / filter_total
                        row['total_ratio_dropped'] = filter_total / total_num_docs
                        rows.append(row)
                    else:
                        print(f"Data for language {lang} does not contain enough items to process.")
                        
            except FileNotFoundError:
                print(f"File not found for language {lang}: {file_path}")
            except json.JSONDecodeError:
                print(f"Error decoding JSON from file for language {lang}: {file_path}")
        

        seen = set()
        ordered_keys = [k for row in rows for k in row if not (k in seen or seen.add(k))]
        writer = csv.DictWriter(file, fieldnames=ordered_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print("Done generating the filter statistics files")

if __name__ == '__main__':
  fire.Fire(main)

