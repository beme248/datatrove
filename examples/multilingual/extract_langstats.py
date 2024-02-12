import json
import random

from datatrove.utils.langstats import compute_wikistats


# Top 10 languages other than English in CommonCrawl according to (https://arxiv.org/pdf/2306.01116.pdf)
# TODO: ja, zh don't have good tokenizers
LANGUAGES = ["ru", "de", "es", "ja", "fr", "zh", "it", "pt", "nl", "pl"]
SAMPLES_PER_LANG = 20000
OUTFILE = f"stats_{SAMPLES_PER_LANG}_{len(LANGUAGES)}.json"

random.seed(42)

language_stats = {}

for language in LANGUAGES:
    print(f"Processing '{language}' language")
    stats = compute_wikistats(language, n_samples=SAMPLES_PER_LANG)
    language_stats[language] = stats

with open(OUTFILE, "w") as f:
    json.dump(language_stats, f)
