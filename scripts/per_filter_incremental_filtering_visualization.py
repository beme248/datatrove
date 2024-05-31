import json
import os
import matplotlib.pyplot as plt
import fire
import numpy as np

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
    "ja"
]

def create_path_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main(
        filter_mode: str = 'wiki',
        dataset_mode: str = 'cc',
        dump: str = "CC-MAIN-2023-23",
        reference_lang: str = 'en'):
    
    output_path = os.path.join("aggregated_filter_statistics")
    create_path_if_not_exists(output_path)

    language_stats = {}
    all_dropped_keys = set()

    # First pass to gather all possible dropped keys
    for lang in languages:
        file_path = os.path.join("processing", f"multilingual_{dataset_mode}_with_{filter_mode}_filters", "logs", dump, lang, "stats.json")
        print(f"path {file_path}")
        try:
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)

                if len(data) > 4:
                    for filter_idx in range(1, 5):
                        filter_stats = data[filter_idx].get("stats", {})
                        for k in filter_stats.keys():
                            if k.startswith("dropped_"):
                                all_dropped_keys.add(k)
        except FileNotFoundError:
            print(f"File not found for language {lang}: {file_path}")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file for language {lang}: {file_path}")

    all_dropped_keys.add('total_dropped')
    breakpoint()
    all_dropped_keys = sorted(all_dropped_keys)  # Sort keys for consistent ordering

    # Second pass to gather statistics and percentages
    for lang in languages:
        file_path = os.path.join("processing", f"multilingual_{dataset_mode}_with_{filter_mode}_filters", "logs", dump, lang, "stats.json")
        print(f"path {file_path}")
        try:
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)

                if len(data) > 4:
                    aggregate_dropped_stats = {}
                    for filter_idx in range(1, 5):
                        filter_stats = data[filter_idx].get("stats", {})
                        for k, v in filter_stats.items():
                            if k.startswith("dropped_"):
                                if k in aggregate_dropped_stats:
                                    aggregate_dropped_stats[k] += v
                                else:
                                    aggregate_dropped_stats[k] = v

                    total_num_docs = data[0].get("stats", {}).get("documents", {}).get("total", 1)  # Default to 1 to avoid division by zero
                    if total_num_docs == 0:
                        total_num_docs = 1  # Avoid division by zero
                    percentage_dropped_stats = {k: (v / total_num_docs) * 100 for k, v in aggregate_dropped_stats.items()}

                    percentage_dropped_stats['total_dropped'] = (total_num_docs - data[-1]['stats']['total']) / total_num_docs

                    language_stats[lang] = percentage_dropped_stats
                else:
                    print(f"Data for language {lang} does not contain enough items to process.")
        except FileNotFoundError:
            print(f"File not found for language {lang}: {file_path}")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file for language {lang}: {file_path}")

    # Load reference statistics
    reference_file_path = os.path.join("processing", f"multilingual_{dataset_mode}_with_default_filters", "logs", dump, reference_lang, "stats.json")
    reference_stats = {}
    try:
        with open(reference_file_path, 'r') as json_file:
            reference_data = json.load(json_file)

            if len(reference_data) > 4:
                for filter_idx in range(1, 5):
                    filter_stats = reference_data[filter_idx].get("stats", {})
                    for k, v in filter_stats.items():
                        if k.startswith("dropped_"):
                            if k in reference_stats:
                                reference_stats[k] += v
                            else:
                                reference_stats[k] = v

                total_num_docs = reference_data[0].get("stats", {}).get("documents", {}).get("total", 1)
                if total_num_docs == 0:
                    total_num_docs = 1
                reference_percentage_stats = {k: (v / total_num_docs) * 100 for k, v in reference_stats.items()}
                reference_percentage_stats['total_dropped'] = (total_num_docs - reference_data[-1]['stats']['total']) / total_num_docs
    except FileNotFoundError:
        print(f"Reference file not found for language {reference_lang}: {reference_file_path}")
        reference_percentage_stats = {}
    except json.JSONDecodeError:
        print(f"Error decoding JSON from reference file for language {reference_lang}: {reference_file_path}")
        reference_percentage_stats = {}

    # Plotting
    num_dropped_keys = len(all_dropped_keys)
    fig, axes = plt.subplots(int((num_dropped_keys + 3) / 4), 4, figsize=(20, 40))  # Adjust the grid to fit all drop types
    axes = axes.flatten()

    colors = plt.cm.tab20(np.linspace(0, 1, 20)).tolist() + plt.cm.tab20b(np.linspace(0, 1, 20)).tolist() + plt.cm.tab20c(np.linspace(0, 1, 20)).tolist()
    colors = colors[:len(all_dropped_keys)]

    for i, drop_type in enumerate(all_dropped_keys):
        percentages = [language_stats[lang].get(drop_type, 0) for lang in languages]
        ax = axes[i]
        ax.bar(languages, percentages, color=colors)
        ax.set_title(drop_type)
        ax.set_xlabel('Languages')
        ax.set_ylabel('Percentage of Dropped Files')
        ax.set_xticklabels(languages, rotation=45, ha='right')
        
        # Plot reference line
        if drop_type in reference_percentage_stats:
            ref_value = reference_percentage_stats[drop_type]
            ax.axhline(y=ref_value, color='red', linestyle='-', linewidth=2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"dropped_files_per_filter_{dataset_mode}_with_{filter_mode}.pdf"), format="pdf")
    plt.show()

if __name__ == '__main__':
    fire.Fire(main)
