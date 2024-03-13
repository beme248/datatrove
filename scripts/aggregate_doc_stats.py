import json
import os

base_path = "/iopsstor/scratch/cscs/bmessmer/fork/datatrove/test/data/datatrove/multi_lingual_stats/base_processing"
LANGUAGES = "af als am an ar arz as ast av az azb ba bar bcl be bg bh bn bo bpy br bs bxr ca cbk ce ceb ckb co cs cv cy da de diq dsb dty dv el eml en eo es et eu fa fi fr frr fy ga gd gl gn gom gu gv he hi hif hr hsb ht hu hy ia id ie ilo io is it ja jbo jv ka kk km kn ko krc ku kv kw ky la lb lez li lmo lo lrc lt lv mai mg mhr min mk ml mn mr mrj ms mt mwl my myv mzn nah nap nds ne new nl nn no oc or os pa pam pfl pl pms pnb ps pt qu rm ro ru rue sa sah sc scn sco sd sh si sk sl so sq sr su sv sw ta te tg th tk tl tr tt tyv ug uk ur uz vec vep vi vls vo wa war wuu xal xmf yi yo yue zh".split(
    " "
)
DUMP = "CC-MAIN-2023-50"

# Iterate through each language and process the JSON file
print(f"================> updated {base_path}")
print(f"Language, Count, MB")
for lang in LANGUAGES[:4]:
    file_path = os.path.join(base_path, lang, "reduce", DUMP, f"{DUMP}.json")
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            if lang in data:
                doc_count = data[lang]["total_docs"]
                num_bytes = data[lang]["total_bytes"]
                print(f"{lang}, {doc_count}, {num_bytes / 1e6}")
            else:
                print(f"Data for language {lang} does not contain enough items to process.")
                
    except FileNotFoundError:
        print(f"File not found for language {lang}: {file_path}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file for language {lang}: {file_path}")