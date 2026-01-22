import json
from pathlib import Path
import sys

import pandas as pd


DATA_FILE = Path("data/raw/dados_status_ocorrencias_gerais.json")
ASSIGNED_CSV = Path("outputs/cvli_with_bairro_municipal.csv")
OUT_FILE = Path("data/raw/dados_status_ocorrencias_gerais_bairros_atribuidos.json")


def load_wrapper(path):
    with open(path, "r", encoding="utf-8") as f:
        wrapper = json.load(f)
    # find data array
    if isinstance(wrapper, dict) and "data" in wrapper:
        return wrapper, wrapper["data"], None
    if isinstance(wrapper, list):
        for i, el in enumerate(wrapper):
            if isinstance(el, dict) and "data" in el:
                return wrapper, el["data"], i
    raise ValueError("Could not find 'data' array in JSON wrapper")


def main():
    if not DATA_FILE.exists():
        print(f"data file not found: {DATA_FILE}")
        sys.exit(1)
    if not ASSIGNED_CSV.exists():
        print(f"assigned CSV not found: {ASSIGNED_CSV}")
        sys.exit(1)

    wrapper, data_arr, wrapper_index = load_wrapper(DATA_FILE)

    assigned = pd.read_csv(ASSIGNED_CSV, dtype=str)
    # ensure id column as string
    if 'id' not in assigned.columns:
        print("assigned CSV missing 'id' column")
        sys.exit(1)

    assigned_map = dict(zip(assigned['id'].astype(str), assigned.get('bairro_assigned', assigned.get('bairro', pd.Series(['']*len(assigned))).fillna(''))))

    updated = 0
    for rec in data_arr:
        rid = rec.get('id')
        if rid is None:
            continue
        sid = str(rid)
        b = assigned_map.get(sid)
        if b and b != 'nan':
            rec['bairro'] = b
            updated += 1

    # write new wrapper file preserving original structure
    with open(OUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(wrapper, f, ensure_ascii=False)

    print(f"Wrote {OUT_FILE} — updated bairro for {updated} records")


if __name__ == '__main__':
    main()
