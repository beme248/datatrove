import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
import yaml
import os

import yaml
from collections import defaultdict
import os
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import matplotlib.font_manager as fm

from matplotlib import font_manager

def is_clean(word):
    word = word.strip()
    return (
        word != "–"
        and word != "—"
        and word != "’"
        and word != "’’"
        and word != "||"
        and word != "|"
        and word != "।"
        and word != "''"
        and word != "'"
        and word != "``"
        and word != "`"
        and word != "‘"
        and word != "„"
        and word != "“"
        and word != "”"
        and word != "«"
        and word != "»"
        and word != "|-"
        and word != ":"
        and word != "："
        and word != "《"
        and word != "》"
        and word != "，"
        and word != "("
        and word != ")"
        and word != "（"
        and word != "）"
        and word != "//"
        and word != "/"
        and word != "\\"
        and word != "\\\\"
        and "=" not in word
        and "\u200d" not in word
        and "align" != word
        and not word.isdigit()
    )

def to_clean(stopwords):
    return [w for w in stopwords if is_clean(w)]

def to_clean_stopwords(lang, word_counter, is_doc_word=False):
    stopwords = to_clean(p_thresh_words(word_counter, 0.008))
    if len(stopwords) < 8 or lang == "sr":
        stopwords = p_thresh_words(word_counter, 0.003)
    if len(stopwords) < 8 and is_doc_word:
        stopwords = p_thresh_words(word_counter, 0.002)
    return stopwords

def p_thresh_words(word_counter, threshold):
    # This function should be implemented to return words based on the given threshold
    # Here is a placeholder implementation
    total_words = sum(word_counter.values())
    threshold_count = total_words * threshold
    return [word for word, count in word_counter.items() if count >= threshold_count]

# List of languages
LANGUAGES = [
    # "en",
    "de", "hr", "pt", "cs", "zh", "fr", "ru", "tr", 
    "ar", "th", "hi", "sw", "te", "ja"
]

# Prepare stopwords data
stopwords_data = {}
for lang in LANGUAGES:
    # with open(f'cc_filters_meanstd/{lang}.yml', 'r') as file:
    with open(f'cc_statistics/{lang}.yml', 'r') as file:
        lang_data = yaml.safe_load(file)
    doc_per_word = lang_data['doc_per_word']
    word_count = lang_data['word_counter']
    
    cc_stopwords = to_clean_stopwords(lang, doc_per_word, is_doc_word=True)
    wiki_stopwords = to_clean_stopwords(lang, word_count)

    with open(f'cc_filters_meanstd/{lang}.yml', 'r') as file:
        lang_data_s = yaml.safe_load(file)
    cc_stopwords_2 = lang_data_s['stopwords'] # to_clean_stopwords(lang, doc_per_word, is_doc_word=True)
    
    stopwords_data[lang] = {
        'stopwords': (set(cc_stopwords), set(wiki_stopwords), set(cc_stopwords_2))
    }

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
plt.rcParams['font.size'] = 10

# Verify font properties being used by Matplotlib
print("Current font properties:", plt.rcParams['font.family'])

# Set the number of rows and columns for the subplots
num_rows = 5
num_cols = 3

# Create a figure with a suitable size to accommodate all subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 5))

# Flatten the axes array for easier indexing
axes = axes.flatten()

plot_idx = 0
for lang, keys_data in stopwords_data.items():
    cc_set, wiki_set, cc_stopwords_2 = keys_data['stopwords']
    if plot_idx >= len(axes):
        break  # Ensure we do not exceed the available subplot slots
    
    assert cc_set == cc_stopwords_2, "different stopwords for filters and statistics"

    ax = axes[plot_idx]
    
    # Create the Venn diagram
    venn = venn2([cc_set, wiki_set], ('Doc Count', 'Word Count'), ax=ax)
    
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
    
    ax.set_title(f'Stopwords for {lang}')

    plot_idx += 1

# Hide any unused subplots
for i in range(plot_idx, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.savefig(os.path.join('aggregated_filter_statistics', 'cc_one_doc_vs_total_languages_debug3.pdf'))
plt.show()
