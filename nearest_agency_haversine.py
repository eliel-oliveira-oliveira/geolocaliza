
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nearest agency by straight-line (haversine) — com suporte a CSV separado por ponto e vírgula.
Uso:
  python nearest_agency_haversine.py --clients clients.csv --agencies agencies.csv --outdir out --sep ";" --geocode-missing
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd

# Optional: geocoding for missing lat/lon by CEP (demo only; not for 340k in prod)
try:
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter
    HAS_GEOPY = True
except Exception:
    HAS_GEOPY = False

try:
    from sklearn.neighbors import BallTree
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

EARTH_RADIUS_M = 6371000.0

def ensure_columns(df, needed, name):
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{name} está faltando colunas obrigatórias: {missing}. Presentes: {list(df.columns)}")

def load_csv(path, name, sep=';'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    df = pd.read_csv(path, dtype=str, sep=sep).replace({np.nan: None})
    # normalizar decimal brasileiro em lat/lon
    for col in ('lat', 'lon'):
        if col in df.columns:
            df[col] = df[col].str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def init_geocoder(user_agent='itau-nearest-agency'):
    if not HAS_GEOPY:
        raise RuntimeError("geopy não está instalado. Instale com: pip install geopy")
    geocoder = Nominatim(user_agent=user_agent, timeout=10)
    from geopy.extra.rate_limiter import RateLimiter
    rate_limited = RateLimiter(geocoder.geocode, min_delay_seconds=1, swallow_exceptions=True)
    return rate_limited

def load_geocode_cache(cache_path):
    if os.path.exists(cache_path):
        cache = pd.read_csv(cache_path, dtype={'cep': str, 'lat': float, 'lon': float})
        cache = cache.dropna(subset=['cep', 'lat', 'lon']).drop_duplicates('cep')
        return cache
    return pd.DataFrame(columns=['cep', 'lat', 'lon'])

def save_geocode_cache(cache_df, cache_path):
    cache_df.to_csv(cache_path, index=False)

def clean_cep(cep):
    if cep is None:
        return None
    s = ''.join([c for c in str(cep) if c.isdigit()])
    return s if len(s) >= 5 else None

def geocode_by_cep(df, cache_path='/mnt/data/geocode_cache.csv', max_rows=None):
    rate_geocode = init_geocoder()
    cache = load_geocode_cache(cache_path)
    cache_map = {row['cep']: (row['lat'], row['lon']) for _, row in cache.iterrows()}

    rows_to_geo = df[(df['lat'].isna()) | (df['lon'].isna())].copy()
    if max_rows is not None:
        rows_to_geo = rows_to_geo.head(max_rows)

    updated = 0
    new_cache = []
    for idx, row in rows_to_geo.iterrows():
        cep = clean_cep(row.get('cep'))
        if not cep:
            continue
        if cep in cache_map:
            lat, lon = cache_map[cep]
        else:
            try:
                result = rate_geocode({'postalcode': cep, 'country': 'BR'})
            except Exception:
                result = None
            if result is None:
                continue
            lat, lon = result.latitude, result.longitude
            cache_map[cep] = (lat, lon)
            new_cache.append({'cep': cep, 'lat': lat, 'lon': lon})
        df.at[idx, 'lat'] = lat
        df.at[idx, 'lon'] = lon
        updated += 1

    if new_cache:
        cache = pd.concat([cache, pd.DataFrame(new_cache)], ignore_index=True).drop_duplicates('cep', keep='last')
        save_geocode_cache(cache, cache_path)
    return df, updated

def nearest_agency(clients_df, agencies_df, outdir, sep=';', distance_buckets_km=(5, 20, 50)):
    if not HAS_SKLEARN:
        raise RuntimeError("scikit-learn não está instalado. Instale com: pip install scikit-learn")

    clients = clients_df.dropna(subset=['lat', 'lon']).copy()
    agencies = agencies_df.dropna(subset=['lat', 'lon']).copy()

    if clients.empty:
        raise ValueError("Nenhum cliente com lat/lon válidos para processar.")
    if agencies.empty:
        raise ValueError("Nenhuma agência com lat/lon válidos para processar.")

    clients_rad = np.radians(clients[['lat', 'lon']].to_numpy(dtype=float))
    agencies_rad = np.radians(agencies[['lat', 'lon']].to_numpy(dtype=float))

    tree = BallTree(agencies_rad, metric='haversine')
    dist_rad, ind = tree.query(clients_rad, k=1)
    dist_m = dist_rad.flatten() * EARTH_RADIUS_M
    dist_km = dist_m / 1000.0
    idx_agencies = ind.flatten()

    nearest = agencies.iloc[idx_agencies].reset_index(drop=True)
    out = pd.DataFrame({
        'client_id': clients['client_id'].values,
        'client_cep': clients['cep'].values if 'cep' in clients.columns else None,
        'agency_id': nearest['agency_id'].values,
        'agency_name': nearest['name'].values if 'name' in nearest.columns else None,
        'agency_city': nearest['city'].values if 'city' in nearest.columns else None,
        'agency_state': nearest['state'].values if 'state' in nearest.columns else None,
        'distance_km': np.round(dist_km, 6)
    })

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, 'nearest_mapping.csv')
    out.to_csv(out_path, index=False, sep=sep)

    # Buckets
    edges = list(distance_buckets_km) + [float('inf')]
    labels = [f"<= {edge:.0f} km" for edge in distance_buckets_km] + [f"> {distance_buckets_km[-1]:.0f} km"]

    counts = np.zeros(len(labels), dtype=int)
    for d in dist_km:
        for i, edge in enumerate(edges):
            if d <= edge:
                counts[i] += 1
                break

    total = len(dist_km)
    pct = (counts / total) * 100.0
    summary = pd.DataFrame({'bucket': labels, 'clients': counts, 'pct': np.round(pct, 2)})
    summary_path = os.path.join(outdir, 'impact_summary.csv')
    summary.to_csv(summary_path, index=False, sep=sep)

    print(f"Processados {total} clientes. Arquivos salvos:")
    print(f" - {out_path}")
    print(f" - {summary_path}")
    return out_path, summary_path

def main():
    parser = argparse.ArgumentParser(description="Nearest Itaú agency por linha reta (haversine).")
    parser.add_argument("--clients", required=True, help="CSV de clientes (client_id, cep, lat, lon)")
    parser.add_argument("--agencies", required=True, help="CSV de agências (agency_id, name, cep, address, city, state, lat, lon)")
    parser.add_argument("--outdir", default="out", help="Diretório de saída")
    parser.add_argument("--sep", default=";", help="Separador de CSV para leitura e escrita (ex.: ';' ou ',')")
    parser.add_argument("--geocode-missing", action="store_true", help="Geocodificar lat/lon faltantes por CEP (DEMO; não usar em 340k em produção)")
    parser.add_argument("--geocode-max", type=int, default=200, help="Máximo de linhas para geocodificar nesta execução (para evitar rate limit)")
    parser.add_argument("--buckets", type=str, default="5,20,50", help="Buckets de distância em km, separados por vírgula")
    args = parser.parse_args()

    clients_df = load_csv(args.clients, "clientes", sep=args.sep)
    agencies_df = load_csv(args.agencies, "agências", sep=args.sep)

    ensure_columns(clients_df, ['client_id', 'cep'], "clientes")
    ensure_columns(agencies_df, ['agency_id', 'cep'], "agências")

    for df in (clients_df, agencies_df):
        if 'lat' not in df.columns:
            df['lat'] = np.nan
        if 'lon' not in df.columns:
            df['lon'] = np.nan

    if args.geocode_missing:
        print("Geocodificando CEPs com lat/lon ausentes (DEMO, lento e com limite de taxa)...")
        clients_df, c_upd = geocode_by_cep(clients_df, max_rows=args.geocode_max)
        agencies_df, a_upd = geocode_by_cep(agencies_df, max_rows=args.geocode_max)
        print(f"Geocodificados (clientes): {c_upd} | (agências): {a_upd}")

    try:
        buckets = tuple(float(x.strip()) for x in args.buckets.split(","))
        buckets = tuple(sorted([b for b in buckets if b > 0]))
    except Exception:
        buckets = (5, 20, 50)

    out_path, summary_path = nearest_agency(clients_df, agencies_df, args.outdir, sep=args.sep, distance_buckets_km=buckets)
    print("Concluído.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERRO] {e}")
        sys.exit(1)
