import yaml
from collections import defaultdict
import os
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import matplotlib.font_manager as fm

from matplotlib import font_manager


# Plot numerical data
plt.rcParams.update({
    'figure.constrained_layout.use': True,
    'lines.linewidth': 1,
    'lines.markersize': 3,
    'font.size': 6,
    'axes.labelsize': 6,
    'legend.fontsize': 6,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
})

default_values = {
    'line_punct_thr' : 0.12,
    'short_line_thr' : 0.67,
    'short_line_length' : 30,
    'char_duplicates_ratio' : 0.01,
    'new_line_ratio' : 0.3,
    'min_doc_words' : 50,
    'max_doc_words' : 100000,
    'min_avg_word_length' : 3,
    'max_avg_word_length' : 10,
    'max_symbol_word_ratio' : 0.1,
    'max_bullet_lines_ratio' : 0.9,
    'max_ellipsis_lines_ratio' : 0.3,
    'max_non_alpha_words_ratio' : 0.8,
    'min_stop_words' : 2,
}

def create_path_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

LANGUAGES = [
    "en", "de", "hr", "pt", "cs", "zh", "fr", "ru", "tr",
    "ar", "th", "hi", "sw", "te", "ja"
]

output_path = os.path.join("aggregated_filter_statistics_v2")
create_path_if_not_exists(output_path)

# Store data for combined plots
combined_data = defaultdict(lambda: defaultdict(list))
stopwords_data = defaultdict(lambda: defaultdict(dict))

for run_type in ["filters_meanstd"]:
    for lang in LANGUAGES:
        cc_path = f"cc_{run_type}/{lang}.yml"
        wiki_path = f"wiki_{run_type}/{lang}.yml"

        with open(cc_path, 'r') as f:
            cc_data = yaml.safe_load(f)

        with open(wiki_path, 'r') as f:
            wiki_data = yaml.safe_load(f)

        keys = set(cc_data.keys()).union(set(wiki_data.keys()))

        for key in keys:
            if key in cc_data and key in wiki_data:
                cc_value = cc_data[key]
                wiki_value = wiki_data[key]
                if isinstance(cc_value, (float, int)) and isinstance(wiki_value, (float, int)):
                    combined_data[key][lang] = [cc_value, wiki_value]
                elif isinstance(cc_value, list) and isinstance(wiki_value, list):
                    stopwords_data[lang][key] = (set(cc_value), set(wiki_value))

num_keys = len(combined_data)
fig = plt.figure(figsize=(6.5, 1.5 * num_keys))

colors = ['#206675', '#ff8214']
labels = ['cc', 'wiki']
for i, (key, lang_data) in enumerate(combined_data.items()):
    plt.subplot(num_keys, 1, i + 1)
    bar_width = 0.35
    index = range(len(lang_data))
    
    for j, (lang, values) in enumerate(lang_data.items()):
        positions = [(x * bar_width) + 2 * (j * bar_width) for x in range(2)]
        plt.bar(positions, values, bar_width, label=f'{lang} {key}', color=colors)
    
    if key in default_values:
        plt.axhline(y=default_values[key], color='red', linestyle='--', linewidth=1)

    plt.title(f'{key}')
    plt.ylabel(key)
    plt.xticks([r * 2 * bar_width for r in range(len(lang_data))], [f'{lang}' for lang in lang_data.keys()], rotation=90)

custom_legend_handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors]
plt.subplot(num_keys, 1, 1).legend(custom_legend_handles, labels, loc='upper right', ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'all_keys_all_languages.pdf'))
plt.close()

# Plot Venn diagrams for stopwords
for font in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
    if 'NotoSans' in font:
        print(font)

font_paths = [
    os.path.expanduser('~/.local/share/fonts/noto/NotoSans-Regular.ttf'),
    os.path.expanduser('~/.local/share/fonts/noto/NotoSansCJKjp-Regular.otf'),
    os.path.expanduser('~/.local/share/fonts/noto/NotoSansTelugu-Regular.ttf'),
    os.path.expanduser('~/.local/share/fonts/noto/NotoSansThai-Regular.ttf'),
    os.path.expanduser('~/.local/share/fonts/noto/NotoSansDevanagari-Regular.ttf'),
    os.path.expanduser('~/.local/share/fonts/noto/NotoSansArabic-Regular.ttf')
]

for font_path in font_paths:
    fm.fontManager.addfont(font_path)


font_properties = [fm.FontProperties(fname=path) for path in font_paths]
font_names = [prop.get_name() for prop in font_properties]
print(f"Font names being used: {font_names}")

plt.rcParams['font.family'] = font_names
plt.rcParams['font.sans-serif'] = font_names
plt.rcParams['font.size'] = 12

# Verify font properties being used by Matplotlib
print("Current font properties:", plt.rcParams['font.family'])

# Set the number of rows and columns for the subplots
num_rows = 5
num_cols = 3

# Create a figure with a suitable size to accommodate all subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 5))

# Flatten the axes array for easier indexing
axes = axes.flatten()

# Plot Venn diagrams for stopwords
plot_idx = 0
for lang, keys_data in stopwords_data.items():
    for key, (cc_set, wiki_set) in keys_data.items():
        if plot_idx >= len(axes):
            break  # Ensure we do not exceed the available subplot slots
        
        ax = axes[plot_idx]
        
        # Create the Venn diagram
        venn = venn2([cc_set, wiki_set], ('CC', 'Wiki'), ax=ax)
        
        # Set colors
        venn.get_patch_by_id('10').set_color('#206675')  # Left circle
        venn.get_patch_by_id('01').set_color('#081b2a')  # Right circle
        venn.get_patch_by_id('11').set_color('#ff8214')  # Intersection
        
        for subset in ('10', '01', '11'):
            if venn.get_label_by_id(subset) is not None:
                if subset == '10':
                    venn.get_label_by_id(subset).set_text('\n'.join(cc_set - wiki_set))
                elif subset == '01':
                    venn.get_label_by_id(subset).set_text('\n'.join(wiki_set - cc_set))
                elif subset == '11':
                    venn.get_label_by_id(subset).set_text('\n'.join(cc_set & wiki_set))
        
        ax.set_title(f'{key} stopwords for {lang}')

        plot_idx += 1

# Hide any unused subplots
for i in range(plot_idx, len(axes)):
    fig.delaxes(axes[i])

# Adjust layout to prevent overlap
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'all_languages_venn.pdf'))
plt.close()

print("Combined plot generated and saved successfully.")
