import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime
import os

try:
    import holidays as _holidays
except Exception:
    _holidays = None


def load_inmet_aggregated(inmet_csv_path, gdf_nodes, date_col='date', lat_col='lat', lon_col='lon'):
    """
    Lê CSV do INMET com colunas de data, lat, lon e variáveis (ex: precip_mm, temp_mean),
    mapeia pontos aos polígonos de `gdf_nodes` e retorna DataFrame agregada por data x node.

    CSV esperado: um ponto por estação por dia ou por horário com colunas parseáveis como float.
    """
    if not os.path.exists(inmet_csv_path):
        return pd.DataFrame()

    df = pd.read_csv(inmet_csv_path)
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce').dt.date
    else:
        # tenta detectar coluna de data
        for c in df.columns:
            if 'date' in c.lower() or 'data' in c.lower():
                df[c] = pd.to_datetime(df[c], errors='coerce').dt.date
                date_col = c
                break

    # cria geometria
    if lat_col not in df.columns or lon_col not in df.columns:
        # tenta colunas alternativas
        for c in df.columns:
            if 'lat' in c.lower() and lat_col not in df.columns:
                lat_col = c
            if 'lon' in c.lower() or 'lng' in c.lower():
                lon_col = c

    df = df.dropna(subset=[lat_col, lon_col, date_col])
    df['lat'] = pd.to_numeric(df[lat_col], errors='coerce')
    df['lon'] = pd.to_numeric(df[lon_col], errors='coerce')
    df = df.dropna(subset=['lat', 'lon'])

    g_stations = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df.lon, df.lat)], crs='EPSG:4326')

    # garante mesmo CRS e realiza spatial join
    if gdf_nodes.crs != g_stations.crs:
        try:
            gdf_nodes = gdf_nodes.to_crs(g_stations.crs)
        except Exception:
            pass

    joined = gpd.sjoin(g_stations, gdf_nodes, how='left', predicate='within')

    # se não houve join por 'within', faz nearest
    if joined['index_right'].isna().all():
        # aproximação por nearest
        g_stations = g_stations.set_geometry('geometry')
        gdf_nodes = gdf_nodes.set_geometry('geometry')
        joined = gpd.sjoin_nearest(g_stations, gdf_nodes, how='left')

    # assume que o gdf_nodes tem coluna 'name' ou equivalente
    node_name_col = 'name' if 'name' in gdf_nodes.columns else gdf_nodes.columns[0]
    joined['node'] = joined[node_name_col].astype(str).str.upper().str.strip()

    # Agrega por data e node
    agg_cols = [c for c in joined.columns if any(k in c.lower() for k in ['precip', 'chuva', 'mm', 'temp', 'temper', 't_mean', 'tmax', 'tmin'])]
    if not agg_cols:
        # nada para agregar
        return pd.DataFrame()

    grouped = joined.groupby([date_col, 'node'])[agg_cols].mean().reset_index()
    grouped = grouped.rename(columns={date_col: 'date'})
    grouped['date'] = pd.to_datetime(grouped['date']).dt.date
    return grouped


def holidays_series(dates, country='BR'):
    """Retorna um dict date -> 1/0 indicando feriado nacional (fallback limitado se pacote inexistente)."""
    out = {}
    if _holidays is not None:
        yrs = set([d.year for d in dates])
        br = _holidays.CountryHoliday(country, years=yrs)
        for d in dates:
            out[d.date()] = 1 if d.date() in br else 0
        return out

    # fallback manual (principais feriados nacionais fixos — não exaustivo)
    fixed = set()
    for d in dates:
        y = d.year
        fixed.update({
            datetime(y,1,1).date(),   # Ano Novo
            datetime(y,5,1).date(),   # Dia do Trabalho
            datetime(y,9,7).date(),   # Independência
            datetime(y,10,12).date(), # Nossa Sra Aparecida
            datetime(y,11,2).date(),  # Finados
            datetime(y,11,15).date(), # Proclamação
            datetime(y,12,25).date()  # Natal
        })
    for d in dates:
        out[d.date()] = 1 if d.date() in fixed else 0
    return out


def build_exog_matrix(dates_index, nodes, inmet_agg_df=None):
    """
    Retorna um DataFrame (date x node x features) no formato (num_days, num_nodes, num_features)
    com colunas ordenadas ['precip_mm','temp_mean','holiday'] quando disponíveis.
    """
    import numpy as np
    num_days = len(dates_index)
    num_nodes = len(nodes)
    features = ['precip_mm', 'temp_mean', 'holiday']
    K = len(features)

    mat = np.zeros((num_days, num_nodes, K), dtype=float)

    # Map nodes para índice
    node_idx = {n: i for i, n in enumerate(nodes)}

    # Holidays
    holidays_map = holidays_series(dates_index)
    for di, d in enumerate(dates_index):
        h = holidays_map.get(d, 0)
        mat[di, :, 2] = h

    # INMET
    if inmet_agg_df is not None and not inmet_agg_df.empty:
        inmet_agg_df['date'] = pd.to_datetime(inmet_agg_df['date']).dt.date
        for _, row in inmet_agg_df.iterrows():
            d = row['date']
            node = str(row['node']).upper().strip()
            if node in node_idx:
                ni = node_idx[node]
                try:
                    di = dates_index.get_loc(pd.Timestamp(d))
                except Exception:
                    # dates_index may be DatetimeIndex of Timestamps; fallback linear search
                    try:
                        di = list(dates_index.date).index(d)
                    except Exception:
                        continue
                # tenta preencher precip/temp se existirem
                if 'precip_mm' in row.index:
                    mat[di, ni, 0] = float(row['precip_mm'] or 0)
                else:
                    # tenta colunas que contenham 'precip'
                    for c in row.index:
                        if 'precip' in c.lower() or 'chuva' in c.lower() or 'mm' in c.lower():
                            val = row[c]
                            mat[di, ni, 0] = float(val or 0)
                            break
                # temp
                for c in row.index:
                    if 'temp' in c.lower() or 't_mean' in c.lower() or 'temper' in c.lower():
                        mat[di, ni, 1] = float(row[c] or 0)
                        break

    return mat, features
