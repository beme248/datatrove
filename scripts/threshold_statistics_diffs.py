import yaml
from collections import defaultdict

LANGUAGES = [
    "en",
    "de",
    "hr",
    # "pt",
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
]

for run_type in [ "filters" ]:
    for lang in LANGUAGES:
        cc_path = f"cc-{run_type}/{lang}.yml"
        wiki_path = f"wiki-{run_type}/{lang}.yml"

        statistic_difference = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        with open(cc_path, 'r') as f:
            cc_data = yaml.safe_load(f)

        with open(wiki_path, 'r') as f:
            wiki_data = yaml.safe_load(f)

        for key, value in wiki_data.items():
            if key in [ 'doc_per_word', 'length_counter', 'word_counter' ]:
                pass 
            elif isinstance(value, str):
                assert wiki_data[key] == cc_data[key]
            elif isinstance(value, float) or isinstance(value, int):
                difference = wiki_data[key] - cc_data[key]
                statistic_difference[key]['absolut_difference'] = difference
                statistic_difference[key]['ratio'] = difference / wiki_data[key]

            elif isinstance(value, dict):
                if value.keys() == ['mean', 'std']:
                    for sub_key in value.keys():
                        difference = wiki_data[key][sub_key] - cc_data[key][sub_key]
                        statistic_difference[key][sub_key]['absolut_difference'] = difference
                        statistic_difference[key][sub_key]['ratio'] = difference / wiki_data[key][sub_key]
            else:
                raise ValueError(f'Missing key comparison: {key}')

yaml.dump('cc_difference.yml')