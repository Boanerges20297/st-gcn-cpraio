import json
from pathlib import Path
import sys

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


DATA_FILE = Path("data/raw/dados_status_ocorrencias_gerais.json")
FORT_BAIRROS = Path("data/graph/fortaleza_bairros.geojson")
MUNICIPIOS = Path("data/graph/municipios_ceara_buffer.geojson")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)


def detect_latlon(df):
    lat_col = None
    lon_col = None
    for c in df.columns:
        lc = c.lower()
        if not lat_col and "lat" in lc:
            lat_col = c
        if not lon_col and ("lon" in lc or "lng" in lc):
            lon_col = c
    return lat_col, lon_col


def detect_name_field(gdf):
    candidates = [
        'name', 'NAME', 'nm_mun', 'NM_MUN', 'NM_MUNIC', 'municipio', 'MUNICIPIO', 'NM_MUN8'
    ]
    for c in candidates:
        if c in gdf.columns:
            return c
    # fallback: pick first non-geometry column
    for c in gdf.columns:
        if c != gdf.geometry.name:
            return c
    return None


def main():
    if not DATA_FILE.exists():
        print(f"data file not found: {DATA_FILE}")
        sys.exit(1)

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        wrapper = json.load(f)

    # The dump may be a dict {'data': [...] } or a list containing a table object
    records = None
    if isinstance(wrapper, dict) and "data" in wrapper:
        records = wrapper["data"]
    elif isinstance(wrapper, list):
        # find first dict element that has a 'data' key
        for el in wrapper:
            if isinstance(el, dict) and "data" in el:
                records = el["data"]
                break

    if records is None:
        print("Unexpected JSON structure: expected top-level 'data' array or a list wrapper containing it")
        sys.exit(1)

    df = pd.json_normalize(records)

    lat_col, lon_col = detect_latlon(df)
    if not lat_col or not lon_col:
        print("Could not find latitude/longitude columns in data")
        sys.exit(1)

    # ensure numeric
    df = df.copy()
    df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
    df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')

    total = len(df)
    original_bairro_non_null = df['bairro'].notna().sum() if 'bairro' in df.columns else 0

    # create points
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326",
    )

    # prepare outputs to add
    gdf['bairro_assigned'] = None
    gdf['bairro_source'] = None
    gdf['municipio_assigned'] = None

    # 1) Fortaleza bairros (fine-grained)
    if FORT_BAIRROS.exists():
        fort = gpd.read_file(FORT_BAIRROS)
        if fort.crs is None:
            fort.set_crs(epsg=4326, inplace=True)
        fort = fort.to_crs(gdf.crs)

        # restrict points to records whose cidade appears to be Fortaleza
        def is_fortaleza(val):
            if val is None: return False
            try:
                return str(val).strip().lower() == 'fortaleza'
            except Exception:
                return False

        mask_fort = gdf['cidade'].apply(is_fortaleza) if 'cidade' in gdf.columns else gpd.pd.Series([False]*len(gdf))
        gdf_fort = gdf[mask_fort].copy()
        if not gdf_fort.empty:
            joined = gpd.sjoin(gdf_fort, fort, how='left', predicate='within')
            # detect name field in fort polygons
            fort_name = detect_name_field(fort)
            if fort_name is None:
                fort_name = 'name'

            for idx, row in joined.iterrows():
                bname = row.get(fort_name)
                if pd.notna(bname):
                    gdf.at[idx, 'bairro_assigned'] = bname
                    gdf.at[idx, 'bairro_source'] = 'spatial_fortaleza'

    # 2) Municipio polygons (IBGE buffer) to assign municipio for all points
    if MUNICIPIOS.exists():
        muns = gpd.read_file(MUNICIPIOS)
        if muns.crs is None:
            muns.set_crs(epsg=4326, inplace=True)
        muns = muns.to_crs(gdf.crs)
        mun_name = detect_name_field(muns)

        joined_mun = gpd.sjoin(gdf, muns[[muns.geometry.name, mun_name]], how='left', predicate='within')
        for idx, row in joined_mun.iterrows():
            mval = row.get(mun_name)
            if pd.notna(mval):
                gdf.at[idx, 'municipio_assigned'] = mval

    # For rows where original bairro exists, prefer it
    if 'bairro' in gdf.columns:
        for idx, row in gdf.iterrows():
            if pd.notna(row.get('bairro')) and not pd.isna(row.get('bairro')):
                gdf.at[idx, 'bairro_assigned'] = row.get('bairro')
                gdf.at[idx, 'bairro_source'] = 'original'

    # summary
    assigned_by_spatial_fortaleza = int((gdf['bairro_source'] == 'spatial_fortaleza').sum())
    assigned_municipio = int(gdf['municipio_assigned'].notna().sum())

    summary = {
        'total_rows': int(total),
        'original_bairro_non_null': int(original_bairro_non_null),
        'assigned_by_spatial_fortaleza': assigned_by_spatial_fortaleza,
        'assigned_municipio': assigned_municipio,
    }

    out_csv = OUT_DIR / 'cvli_with_bairro_municipal.csv'
    # drop geometry for csv output
    out_df = gdf.drop(columns=[gdf.geometry.name])
    out_df.to_csv(out_csv, index=False)

    (OUT_DIR / 'bairro_assignment_summary_municipal.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    # sample assigned rows
    sample = out_df[out_df['bairro_source'].notna()].head(50).to_dict(orient='records')
    (OUT_DIR / 'bairro_assigned_sample_municipal.json').write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f"Wrote {out_csv}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
