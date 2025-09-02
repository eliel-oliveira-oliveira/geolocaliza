
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nearest agency by straight-line (haversine) — com suporte a CSV separado por ponto e vírgula.

Este script encontra, para cada cliente, a agência mais próxima em linha reta (distância esférica
no globo) usando a métrica de Haversine. Ele lê CSVs (clientes e agências), calcula as distâncias,
gera um mapeamento cliente→agência e um resumo de impacto por faixas de distância.

Principais decisões de design:
- Usamos BallTree (scikit-learn) com métrica Haversine para consultas de vizinho mais próximo (k=1),
  o que é muito eficiente mesmo com dezenas/centenas de milhares de pontos.
- Mantemos a opção de geocodificar CEPs faltantes (apenas DEMO). Para bases grandes, recomenda-se
  geocodificação em lote via provedores adequados (Google/Mapbox/HERE).
- Suporte a CSV com separador configurável (padrão ';') e normalização de decimais com vírgula.
- Exporta dois arquivos: nearest_mapping.csv e impact_summary.csv.
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd

# Geocodificação (opcional) — somente para demonstração / poucas linhas.
try:
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter
    HAS_GEOPY = True
except Exception:
    HAS_GEOPY = False

# Estrutura de vizinhança para Haversine (rápido e escalável).
try:
    from sklearn.neighbors import BallTree
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

# Raio médio da Terra em metros (usado para converter distância angular em linear).
EARTH_RADIUS_M = 6371000.0


def ensure_columns(df: pd.DataFrame, needed, name: str) -> None:
    """
    Garante que o DataFrame possua as colunas obrigatórias.

    Parâmetros
    ----------
    df : pd.DataFrame
        Tabela a validar.
    needed : list[str]
        Lista de nomes de colunas obrigatórias.
    name : str
        Rótulo da tabela (usado na mensagem de erro).

    Lógica
    ------
    - Verifica a presença de cada coluna exigida.
    - Se faltar algo, lança um ValueError com detalhes.
    """
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{name} está faltando colunas obrigatórias: {missing}. Presentes: {list(df.columns)}")


def load_csv(path: str, name: str, sep: str = ';') -> pd.DataFrame:
    """
    Lê um CSV no caminho indicado e normaliza lat/lon caso existam.

    Parâmetros
    ----------
    path : str
        Caminho do arquivo CSV.
    name : str
        Nome lógico da tabela (apenas para mensagens).
    sep : str, default ';'
        Separador usado no CSV (ex.: ';' para padrão BR, ',' para padrão US).

    Lógica
    ------
    - Lê o CSV como texto para preservar formatos de CEPs (com zeros à esquerda).
    - Substitui NaN por None para facilitar verificações posteriores.
    - Se existirem colunas 'lat' e/ou 'lon', converte decimais com vírgula para ponto
      e transforma em numérico (valores inválidos viram NaN).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    df = pd.read_csv(path, dtype=str, sep=sep).replace({np.nan: None})
    # Normaliza formato brasileiro de decimal nas colunas de coordenadas
    for col in ('lat', 'lon'):
        if col in df.columns:
            # Troca vírgula por ponto (ex.: "-8,0476" -> "-8.0476")
            df[col] = df[col].str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def init_geocoder(user_agent: str = 'itau-nearest-agency'):
    """
    Inicializa um geocodificador Nominatim (OpenStreetMap) com rate limit simples.

    Observação
    ----------
    - Nominatim possui limites estritos de uso e não é apropriado para 340k registros.
      Utilize um provedor pago/empresarial para produção.
    """
    if not HAS_GEOPY:
        raise RuntimeError("geopy não está instalado. Instale com: pip install geopy")
    geocoder = Nominatim(user_agent=user_agent, timeout=10)
    rate_limited = RateLimiter(geocoder.geocode, min_delay_seconds=1, swallow_exceptions=True)
    return rate_limited


def load_geocode_cache(cache_path: str) -> pd.DataFrame:
    """
    Carrega cache local de geocodificação (para evitar consultas repetidas).

    Lógica
    ------
    - Se existir arquivo CSV de cache, lê e mantém apenas linhas válidas (cep, lat, lon).
    - Remove duplicados por CEP, preservando a última ocorrência.
    """
    if os.path.exists(cache_path):
        cache = pd.read_csv(cache_path, dtype={'cep': str, 'lat': float, 'lon': float})
        cache = cache.dropna(subset=['cep', 'lat', 'lon']).drop_duplicates('cep')
        return cache
    return pd.DataFrame(columns=['cep', 'lat', 'lon'])


def save_geocode_cache(cache_df: pd.DataFrame, cache_path: str) -> None:
    """
    Persiste o cache de geocodificação em disco.
    """
    cache_df.to_csv(cache_path, index=False)


def clean_cep(cep: str) -> str | None:
    """
    Normaliza CEP: remove caracteres não numéricos e garante tamanho mínimo (>=5).

    Motivo
    ------
    - Provedores de geocodificação normalmente esperam apenas dígitos para consulta.
    """
    if cep is None:
        return None
    s = ''.join([c for c in str(cep) if c.isdigit()])
    return s if len(s) >= 5 else None


def geocode_by_cep(df: pd.DataFrame, cache_path: str = '/mnt/data/geocode_cache.csv', max_rows: int | None = None):
    """
    Geocodifica linhas com lat/lon ausentes usando CEP (DEMO!).

    Parâmetros
    ----------
    df : pd.DataFrame
        Tabela (clientes ou agências) contendo coluna 'cep' e possivelmente 'lat'/'lon' faltantes.
    cache_path : str
        Caminho do CSV de cache para reaproveitar CEPs já resolvidos.
    max_rows : int | None
        Limita o número de registros geocodificados nesta execução (para não esbarrar em rate limits).

    Lógica
    ------
    - Carrega/constroi um dicionário CEP→(lat,lon) a partir do cache.
    - Seleciona apenas as linhas que precisam de geocodificação (lat/lon ausentes).
    - Para cada CEP:
        * Normaliza o valor (apenas dígitos).
        * Se já existir no cache, reutiliza coordenadas.
        * Caso contrário, consulta o serviço (Nominatim) e armazena no cache.
    - Atualiza o DataFrame original com as coordenadas encontradas.
    - Salva o cache se houver novidades.
    """
    rate_geocode = init_geocoder()
    cache = load_geocode_cache(cache_path)
    cache_map = {row['cep']: (row['lat'], row['lon']) for _, row in cache.iterrows()}

    # Filtra somente registros que precisam de lat/lon
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
            # Reuso de coordenadas já resolvidas anteriormente
            lat, lon = cache_map[cep]
        else:
            # Consulta externa ao Nominatim — sujeito a limites de taxa
            try:
                result = rate_geocode({'postalcode': cep, 'country': 'BR'})
            except Exception:
                result = None
            if result is None:
                continue
            lat, lon = result.latitude, result.longitude
            cache_map[cep] = (lat, lon)
            new_cache.append({'cep': cep, 'lat': lat, 'lon': lon})

        # Persiste coordenadas no DataFrame (mesma linha original)
        df.at[idx, 'lat'] = lat
        df.at[idx, 'lon'] = lon
        updated += 1

    # Se houve novas resoluções, atualiza e salva o cache em disco
    if new_cache:
        cache = pd.concat([cache, pd.DataFrame(new_cache)], ignore_index=True).drop_duplicates('cep', keep='last')
        save_geocode_cache(cache, cache_path)
    return df, updated


def nearest_agency(
    clients_df: pd.DataFrame,
    agencies_df: pd.DataFrame,
    outdir: str,
    sep: str = ';',
    distance_buckets_km: tuple[float, ...] = (5, 20, 50),
):
    """
    Calcula a agência mais próxima para cada cliente usando distância em linha reta (Haversine).

    Parâmetros
    ----------
    clients_df : pd.DataFrame
        Tabela de clientes com colunas: client_id, cep, lat, lon (lat/lon obrigatórios para cálculo).
    agencies_df : pd.DataFrame
        Tabela de agências com colunas: agency_id, cep, name, city, state, lat, lon.
    outdir : str
        Diretório onde os arquivos de saída serão gravados.
    sep : str, default ';'
        Separador para escrita dos CSVs de saída.
    distance_buckets_km : tuple[float, ...]
        Quebras de distância (em km) para sumarização de impacto.

    Lógica
    ------
    1) Filtra apenas linhas com coordenadas válidas (descarta NaN).
    2) Converte lat/lon para radianos (exigência da BallTree com métrica Haversine).
    3) Constrói uma BallTree com as agências (pontos de referência).
    4) Para cada cliente, consulta o vizinho mais próximo (k=1) na árvore.
       - O retorno são distâncias angulares (radianos) e índices de linha das agências.
       - Converte radianos → metros → quilômetros.
    5) Monta um DataFrame de saída com o pareamento cliente→agência e a distância.
    6) Gera um resumo de impacto por faixas (buckets) de distância.
    7) Escreve ambos os CSVs no diretório de saída.
    """
    if not HAS_SKLEARN:
        raise RuntimeError("scikit-learn não está instalado. Instale com: pip install scikit-learn")

    # 1) Garante apenas linhas com coordenadas numéricas
    clients = clients_df.dropna(subset=['lat', 'lon']).copy()
    agencies = agencies_df.dropna(subset=['lat', 'lon']).copy()

    if clients.empty:
        raise ValueError("Nenhum cliente com lat/lon válidos para processar.")
    if agencies.empty:
        raise ValueError("Nenhuma agência com lat/lon válidos para processar.")

    # 2) Converte coordenadas para radianos (formato esperado pela BallTree/Haversine)
    clients_rad = np.radians(clients[['lat', 'lon']].to_numpy(dtype=float))
    agencies_rad = np.radians(agencies[['lat', 'lon']].to_numpy(dtype=float))

    # 3) Constrói a BallTree usando métrica haversine (distância angular na esfera)
    tree = BallTree(agencies_rad, metric='haversine')

    # 4) Consulta o vizinho mais próximo (k=1) para cada cliente
    dist_rad, ind = tree.query(clients_rad, k=1)  # distâncias em radianos e índices da agência mais próxima
    dist_m = dist_rad.flatten() * EARTH_RADIUS_M  # radianos → metros
    dist_km = dist_m / 1000.0                     # metros → km (float)
    idx_agencies = ind.flatten()                  # vetor 1D com índices de agência

    # 5) Prepara o DataFrame final com mapeamento cliente→agência e distância
    nearest = agencies.iloc[idx_agencies].reset_index(drop=True)
    out = pd.DataFrame({
        'client_id': clients['client_id'].values,
        'client_cep': clients['cep'].values if 'cep' in clients.columns else None,
        'agency_id': nearest['agency_id'].values,
        'agency_name': nearest['name'].values if 'name' in nearest.columns else None,
        'agency_city': nearest['city'].values if 'city' in nearest.columns else None,
        'agency_state': nearest['state'].values if 'state' in nearest.columns else None,
        'distance_km': np.round(dist_km, 6)  # arredondado a 6 casas (precisão suficiente para km)
    })

    # 6) Sumarização por faixas de distância (impacto operacional)
    #    Ex.: (5,20,50) gera os rótulos "<= 5 km", "<= 20 km", "<= 50 km" e "> 50 km".
    edges = list(distance_buckets_km) + [float('inf')]
    labels = [f"<= {edge:.0f} km" for edge in distance_buckets_km] + [f"> {distance_buckets_km[-1]:.0f} km"]

    counts = np.zeros(len(labels), dtype=int)
    for d in dist_km:
        # Encaixa cada distância no primeiro limite superior que couber
        for i, edge in enumerate(edges):
            if d <= edge:
                counts[i] += 1
                break

    total = len(dist_km)
    pct = (counts / total) * 100.0
    summary = pd.DataFrame({'bucket': labels, 'clients': counts, 'pct': np.round(pct, 2)})

    # 7) Escrita dos resultados
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, 'nearest_mapping.csv')
    summary_path = os.path.join(outdir, 'impact_summary.csv')
    out.to_csv(out_path, index=False, sep=sep)
    summary.to_csv(summary_path, index=False, sep=sep)

    print(f"Processados {total} clientes. Arquivos salvos:")
    print(f" - {out_path}")
    print(f" - {summary_path}")
    return out_path, summary_path


def main():
    """
    Ponto de entrada via CLI.

    Lógica
    ------
    - Faz o parsing dos argumentos da linha de comando.
    - Carrega CSVs (respeitando o separador solicitado).
    - Valida colunas mínimas.
    - Garante presença de colunas lat/lon (mesmo que vazias) para fluxo de geocodificação.
    - Opcionalmente, geocodifica CEPs sem coordenadas (DEMO).
    - Converte string de buckets em tupla de floats ordenados.
    - Chama a função principal `nearest_agency` e imprime o caminho dos arquivos de saída.
    """
    parser = argparse.ArgumentParser(description="Nearest Itaú agency por linha reta (haversine).")
    parser.add_argument("--clients", required=True, help="CSV de clientes (client_id, cep, lat, lon)")
    parser.add_argument("--agencies", required=True, help="CSV de agências (agency_id, name, cep, address, city, state, lat, lon)")
    parser.add_argument("--outdir", default="out", help="Diretório de saída")
    parser.add_argument("--sep", default=";", help="Separador de CSV para leitura e escrita (ex.: ';' ou ',')")
    parser.add_argument("--geocode-missing", action="store_true", help="Geocodificar lat/lon faltantes por CEP (DEMO; não usar em 340k em produção)")
    parser.add_argument("--geocode-max", type=int, default=200, help="Máximo de linhas para geocodificar nesta execução (para evitar rate limit)")
    parser.add_argument("--buckets", type=str, default="5,20,50", help="Buckets de distância em km, separados por vírgula")
    args = parser.parse_args()

    # Leitura dos CSVs com separador configurável
    clients_df = load_csv(args.clients, "clientes", sep=args.sep)
    agencies_df = load_csv(args.agencies, "agências", sep=args.sep)

    # Validação de colunas mínimas
    ensure_columns(clients_df, ['client_id', 'cep'], "clientes")
    ensure_columns(agencies_df, ['agency_id', 'cep'], "agências")

    # Garante que existam colunas lat/lon (mesmo que vazias), útil para o fluxo de geocodificação
    for df in (clients_df, agencies_df):
        if 'lat' not in df.columns:
            df['lat'] = np.nan
        if 'lon' not in df.columns:
            df['lon'] = np.nan

    # Geocodificação opcional (DEMO) — ideal apenas para amostras pequenas
    if args.geocode_missing:
        print("Geocodificando CEPs com lat/lon ausentes (DEMO, lento e com limite de taxa)...")
        clients_df, c_upd = geocode_by_cep(clients_df, max_rows=args.geocode_max)
        agencies_df, a_upd = geocode_by_cep(agencies_df, max_rows=args.geocode_max)
        print(f"Geocodificados (clientes): {c_upd} | (agências): {a_upd}")

    # Parse dos buckets informados como string ("5,20,50" → (5.0, 20.0, 50.0))
    try:
        buckets = tuple(float(x.strip()) for x in args.buckets.split(","))
        buckets = tuple(sorted([b for b in buckets if b > 0]))
    except Exception:
        buckets = (5, 20, 50)

    # Execução principal
    out_path, summary_path = nearest_agency(
        clients_df, agencies_df, args.outdir, sep=args.sep, distance_buckets_km=buckets
    )
    print("Concluído.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Em caso de erro, exibimos a mensagem e retornamos código de saída != 0 para sinalizar falha no shell/CI.
        print(f"[ERRO] {e}")
        sys.exit(1)
