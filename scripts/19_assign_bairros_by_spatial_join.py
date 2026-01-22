import json
import os
import sys
from pathlib import Path

try:
    import pandas as pd
    import geopandas as gpd
    from shapely.geometry import Point
except Exception as e:
    print("Missing dependency:", e)
    print("Install geopandas and pandas in your environment and re-run.")
    sys.exit(1)


ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def load_data_records(path):
    with open(path, "r", encoding="utf-8") as f:
        top = json.load(f)
    # find the table object with name dados_status and a data array
    for obj in top:
        if obj.get("type") == "table" and obj.get("name") == "dados_status":
            return obj.get("data", [])
    # fallback: if any object has key 'data' and list, use it
    for obj in top:
        if isinstance(obj, dict) and isinstance(obj.get("data"), list):
            return obj.get("data")
    raise ValueError("data array not found in JSON wrapper")


def main():
    src = RAW / "dados_status_ocorrencias_gerais.json"
    if not src.exists():
        print("Source file not found:", src)
        sys.exit(1)

    records = load_data_records(str(src))
    df = pd.DataFrame(records)

    # Normalize coordinate columns
    df["latitude"] = pd.to_numeric(df.get("latitude"), errors="coerce")
    df["longitude"] = pd.to_numeric(df.get("longitude"), errors="coerce")

    df["cidade_norm"] = df.get("cidade", "").astype(str).str.strip().str.lower()

    # Prepare result columns
    df["bairro_assigned"] = df.get("bairro")
    df["bairro_source"] = df.apply(lambda r: "original" if r.get("bairro") else None, axis=1)

    # Load Fortaleza bairros polygons
    fort_file = ROOT / "data" / "graph" / "fortaleza_bairros.geojson"
    if not fort_file.exists():
        print("Fortaleza bairros file not found:", fort_file)
        print("Cannot assign bairros by spatial join for Fortaleza.")
    else:
        fort_bairros = gpd.read_file(fort_file)
        # ensure consistent column name
        if "name" not in fort_bairros.columns:
            # try common alternatives
            candidates = [c for c in fort_bairros.columns if c.lower().startswith("nome") or c.lower().startswith("name")]
            if candidates:
                fort_bairros = fort_bairros.rename(columns={candidates[0]: "name"})
        fort_bairros["name"] = fort_bairros["name"].astype(str)

        # Select records in Fortaleza with valid coords and no original bairro
        mask = (df["cidade_norm"] == "fortaleza") & df["latitude"].notna() & df["longitude"].notna() & df["bairro_assigned"].isna()
        if mask.sum() > 0:
            pts = gpd.GeoDataFrame(df[mask].copy(), geometry=gpd.points_from_xy(df.loc[mask, "longitude"], df.loc[mask, "latitude"]), crs="EPSG:4326")
            # ensure same CRS
            if fort_bairros.crs is None:
                fort_bairros.set_crs(epsg=4326, inplace=True)
            else:
                fort_bairros = fort_bairros.to_crs(epsg=4326)

            joined = gpd.sjoin(pts, fort_bairros[["name", "geometry"]], how="left", predicate="within")
            # assign bairro where found
            for idx, row in joined.iterrows():
                orig_idx = int(row.name)
                assigned = row.get("name")
                if pd.notna(assigned):
                    df.at[orig_idx, "bairro_assigned"] = assigned.title()
                    df.at[orig_idx, "bairro_source"] = "spatial_fortaleza"

    # Final outputs
    out_csv = OUT / "cvli_with_bairro.csv"
    df.to_csv(out_csv, index=False)

    # summary of assignment
    summary = {
        "total_rows": int(len(df)),
        "with_original_bairro": int(df[df["bairro_source"] == "original"].shape[0]),
        "assigned_by_spatial_fortaleza": int(df[df["bairro_source"] == "spatial_fortaleza"].shape[0]),
    }
    with open(OUT / "bairro_assignment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # save a small sample of assigned Fortaleza rows
    sample = df[df["bairro_source"] == "spatial_fortaleza"].head(50)
    sample.to_json(OUT / "bairro_assigned_sample.json", orient="records", force_ascii=False)

    print("Wrote:")
    print(" -", out_csv)
    print(" -", OUT / "bairro_assignment_summary.json")
    print(" -", OUT / "bairro_assigned_sample.json")


if __name__ == "__main__":
    main()
