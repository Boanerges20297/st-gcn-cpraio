"""
Normaliza bairros no arquivo CSV usando fuzzy matching com 50% threshold
Sem dependência de módulos, usando difflib diretamente
"""

import pandas as pd
import json
from pathlib import Path
from difflib import SequenceMatcher

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ═══════════════════════════════════════════════════════════════════════

INPUT_CSV = Path("data/raw/View_Ocorrencias_Operacionais_Modelo.csv")
OUTPUT_CSV = Path("data/raw/View_Ocorrencias_Operacionais_Modelo_NORMALIZADO.csv")
GEOJSON_PATH = Path("data/graph/fortaleza_bairros.geojson")
REPORT_PATH = Path("outputs/normalizacao_bairros_report.json")

SIMILARITY_THRESHOLD = 0.5  # 50%

# ═══════════════════════════════════════════════════════════════════════
# FUNÇÕES
# ═══════════════════════════════════════════════════════════════════════

def load_official_neighborhoods(geojson_path):
    """Carrega bairros oficiais do GeoJSON."""
    print(f"[*] Carregando bairros oficiais de {geojson_path}...")
    
    try:
        with open(geojson_path, 'r', encoding='utf-8') as f:
            geojson = json.load(f)
        
        neighborhoods = set()
        for feature in geojson.get('features', []):
            props = feature.get('properties', {})
            name = props.get('name', '').strip().upper()
            if name:
                neighborhoods.add(name)
        
        print(f"✅ Carregados {len(neighborhoods)} bairros oficiais")
        return neighborhoods
    
    except Exception as e:
        print(f"❌ Erro ao carregar GeoJSON: {e}")
        return set()

def calculate_similarity(str1, str2):
    """Calcula similaridade entre dois textos (0-1)."""
    if not str1 or not str2:
        return 0.0
    
    s1 = str(str1).strip().upper()
    s2 = str(str2).strip().upper()
    
    if s1 == s2:
        return 1.0
    
    return SequenceMatcher(None, s1, s2).ratio()

def find_best_match(raw_name, official_neighborhoods, threshold=0.5):
    """Encontra melhor match para um bairro raw."""
    if not raw_name or pd.isna(raw_name):
        return None, 0.0
    
    raw_clean = str(raw_name).strip().upper()
    
    # Match exato
    if raw_clean in official_neighborhoods:
        return raw_clean, 1.0
    
    # Fuzzy match
    best_match = None
    best_score = 0.0
    
    for official in official_neighborhoods:
        score = calculate_similarity(raw_clean, official)
        if score > best_score:
            best_score = score
            best_match = official
    
    # Retornar só se passou threshold
    if best_score >= threshold:
        return best_match, best_score
    
    return None, 0.0

# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*80)
    print("NORMALIZAÇÃO DE BAIRROS - CSV OPERAÇÕES POLICIAIS")
    print(f"Threshold: {SIMILARITY_THRESHOLD*100:.0f}%")
    print("="*80)
    
    # 1. Carregar CSV
    print(f"\n[1] Carregando arquivo CSV...")
    if not INPUT_CSV.exists():
        print(f"❌ Arquivo não encontrado: {INPUT_CSV}")
        return False
    
    try:
        df = pd.read_csv(INPUT_CSV, on_bad_lines='skip', encoding='utf-8')
        print(f"✅ Carregados {len(df)} registros")
        print(f"✅ Bairros únicos (antes): {df['BairroOcor'].nunique()}")
    except Exception as e:
        print(f"❌ Erro ao carregar CSV: {e}")
        return False
    
    # 2. Carregar bairros oficiais
    print(f"\n[2] Carregando bairros oficiais...")
    official_neighborhoods = load_official_neighborhoods(GEOJSON_PATH)
    if not official_neighborhoods:
        print(f"❌ Nenhum bairro oficial carregado")
        return False
    
    # 3. Aplicar fuzzy matching
    print(f"\n[3] Aplicando fuzzy matching (threshold: {SIMILARITY_THRESHOLD*100:.0f}%)...")
    
    mapping = {}  # raw_name -> (official_name, score)
    bairros_raw = df['BairroOcor'].unique()
    
    for bairro_raw in bairros_raw:
        if pd.isna(bairro_raw) or not bairro_raw:
            mapping[bairro_raw] = ('DESCONHECIDO', 0.0)
        else:
            best_match, score = find_best_match(bairro_raw, official_neighborhoods, SIMILARITY_THRESHOLD)
            if best_match:
                mapping[bairro_raw] = (best_match, score)
            else:
                mapping[bairro_raw] = (str(bairro_raw).strip().upper(), 0.0)  # Manter original se não houver match
    
    # 4. Aplicar mapeamento
    print(f"✅ Mapeamento criado para {len(mapping)} bairros únicos")
    
    matched_count = sum(1 for _, (_, score) in mapping.items() if score >= SIMILARITY_THRESHOLD)
    print(f"✅ Bairros com match >= {SIMILARITY_THRESHOLD*100:.0f}%: {matched_count}/{len(mapping)}")
    
    df['BairroOcor'] = df['BairroOcor'].map(lambda x: mapping.get(x, (str(x).upper(), 0.0))[0])
    
    print(f"✅ Bairros únicos (depois): {df['BairroOcor'].nunique()}")
    
    # 5. Salvar
    print(f"\n[4] Salvando arquivo normalizado...")
    try:
        df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
        print(f"✅ Arquivo salvo: {OUTPUT_CSV}")
    except Exception as e:
        print(f"❌ Erro ao salvar: {e}")
        return False
    
    # 6. Salvar relatório
    print(f"\n[5] Salvando relatório de normalização...")
    report = {
        'total_registros': len(df),
        'bairros_unicos_antes': len(bairros_raw),
        'bairros_unicos_depois': df['BairroOcor'].nunique(),
        'threshold': SIMILARITY_THRESHOLD,
        'matched_count': matched_count,
        'mapping': {
            str(raw): {
                'matched': official,
                'score': float(score)
            }
            for raw, (official, score) in mapping.items()
        }
    }
    
    try:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(REPORT_PATH, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"✅ Relatório salvo: {REPORT_PATH}")
    except Exception as e:
        print(f"⚠️ Aviso ao salvar relatório: {e}")
    
    # 7. Estatísticas
    print(f"\n[6] ESTATÍSTICAS FINAIS")
    print(f"   Registros totais: {len(df)}")
    print(f"   Bairros únicos (normalizado): {df['BairroOcor'].nunique()}")
    print(f"   Taxa de match: {matched_count/len(mapping)*100:.1f}%")
    
    # Top 10 bairros
    print(f"\n   Top 10 bairros (por quantidade):")
    top10 = df['BairroOcor'].value_counts().head(10)
    for bairro, count in top10.items():
        print(f"      {bairro}: {count} ocorrências")
    
    # Sample
    print(f"\n[7] SAMPLE (primeiros 10 registros):")
    print(df[['Data', 'BairroOcor', 'CidadeOcor', 'Natureza']].head(10).to_string())
    
    print(f"\n✅ NORMALIZAÇÃO CONCLUÍDA COM SUCESSO!")
    print(f"   Input:  {INPUT_CSV}")
    print(f"   Output: {OUTPUT_CSV}")
    print(f"   Report: {REPORT_PATH}")
    
    return True

if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
