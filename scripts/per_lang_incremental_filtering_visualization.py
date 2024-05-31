import json
import os
import matplotlib.pyplot as plt
import fire
from matplotlib.patches import Patch

import numpy as np

languages = [
    "en", "de", "hr", "pt", "cs", "zh",
    "fr", "ru", "tr", "ar", "th", "hi",
    "sw", "te", "ja"
]

def create_path_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(filter_mode, dataset_mode, dump, lang):
    file_path = os.path.join("processing", f"multilingual_{dataset_mode}_with_{filter_mode}_filters", "logs", dump, lang, "stats.json")
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            return data
    except FileNotFoundError:
        print(f"File not found for language {lang}: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file for language {lang}: {file_path}")
        return None

def main(dump: str = "CC-MAIN-2023-23"):
    output_path = os.path.join("aggregated_filter_statistics")
    create_path_if_not_exists(output_path)

    language_stats_cc = {}
    language_stats_wiki = {}
    all_dropped_keys = set()

    # Load data for both filter modes and collect all possible dropped keys
    for lang in languages:
        data_cc = load_data('cc', 'cc', dump, lang)
        data_wiki = load_data('wiki', 'cc', dump, lang)

        if data_cc and len(data_cc) > 4:
            for filter_idx in range(1, 3):
                filter_stats = data_cc[filter_idx].get("stats", {})
                for k in filter_stats.keys():
                    if k.startswith("dropped_"):
                        all_dropped_keys.add(k)
        if data_wiki and len(data_wiki) > 4:
            for filter_idx in range(1, 3):
                filter_stats = data_wiki[filter_idx].get("stats", {})
                for k in filter_stats.keys():
                    if k.startswith("dropped_"):
                        all_dropped_keys.add(k)

    all_dropped_keys = sorted(all_dropped_keys)

    # Aggregate statistics for both filter modes
    for lang in languages:
        data_cc = load_data('cc', 'cc', dump, lang)
        data_wiki = load_data('wiki', 'cc', dump, lang)

        if data_cc and len(data_cc) > 4:
            aggregate_dropped_stats = {}
            for filter_idx in range(1, 5):
                filter_stats = data_cc[filter_idx].get("stats", {})
                for k, v in filter_stats.items():
                    if k.startswith("dropped_"):
                        if k in aggregate_dropped_stats:
                            aggregate_dropped_stats[k] += v
                        else:
                            aggregate_dropped_stats[k] = v

            total_num_docs = data_wiki[0].get("stats", {}).get("documents", {}).get("total", 1)
            if total_num_docs == 0:
                total_num_docs = 1
            percentage_dropped_stats = {k: (v / total_num_docs) * 100 for k, v in aggregate_dropped_stats.items()}
            language_stats_cc[lang] = percentage_dropped_stats

        if data_wiki and len(data_wiki) > 4:
            aggregate_dropped_stats = {}
            for filter_idx in range(1, 5):
                filter_stats = data_wiki[filter_idx].get("stats", {})
                for k, v in filter_stats.items():
                    if k.startswith("dropped_"):
                        if k in aggregate_dropped_stats:
                            aggregate_dropped_stats[k] += v
                        else:
                            aggregate_dropped_stats[k] = v

            total_num_docs = data_wiki[0].get("stats", {}).get("documents", {}).get("total", 1)
            if total_num_docs == 0:
                total_num_docs = 1
            percentage_dropped_stats = {k: (v / total_num_docs) * 100 for k, v in aggregate_dropped_stats.items()}
            language_stats_wiki[lang] = percentage_dropped_stats

    # Plotting
    num_languages = len(languages)
    fig, axes = plt.subplots(5, 3, figsize=(15, 20))
    axes = axes.flatten()
    bar_width = 0.35

    colors_tab20 = plt.cm.tab20(np.linspace(0, 1, 20))
    colors_tab20c = plt.cm.tab20c(np.linspace(0, 1, 20))
    base_colors = np.vstack((colors_tab20, colors_tab20c))
    colors_wiki = base_colors

    def darken_color(color, factor=0.5):
        # return tuple(max(min(1, c * factor), 0) for c in color)
        return (0.75, 0.75, 0.75)
    colors_cc = [darken_color(c) for c in base_colors]

    print(f"LEN: {len(base_colors)}, {len(all_dropped_keys)}")

    for i, lang in enumerate(languages):
        stats_wiki = language_stats_wiki.get(lang, {})
        stats_cc = language_stats_cc.get(lang, {})
        percentages_wiki = [stats_wiki.get(k, 0) for k in all_dropped_keys]
        percentages_cc = [stats_cc.get(k, 0) for k in all_dropped_keys]

        ax = axes[i]
        indices = np.arange(len(all_dropped_keys))

        bars_wiki = ax.bar(indices - bar_width / 2, percentages_wiki, bar_width, label='Wiki', color=colors_wiki)
        ax.bar(indices + bar_width / 2, percentages_cc, bar_width, label='CC', color=colors_cc)
        
        if i == 0:
            all_bars_wiki = bars_wiki

        ax.set_title(f"{lang}")
        ax.set_xlabel('Categories')
        ax.set_ylabel('Percentage of Dropped Files')
        ax.set_xticks(indices)
        ax.set_xticklabels([])

    cc_patch = Patch(color=colors_cc[0], label='Wiki Filters')
    fig.legend(list(all_bars_wiki) + [cc_patch], list(all_dropped_keys) + ['CC Filters'], loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02))
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.savefig(os.path.join(output_path, f"dropped_files_per_lang_cc_vs_wiki.pdf"), format="pdf")
    plt.show()

if __name__ == '__main__':
    import fire
    fire.Fire(main)
