"""
Cruzamento de Dados: Sazonalidade CVLI x Prisões (Operações RAIO)
Objetivo: Identificar padrões de correlação, diminuição ou aumento forte
entre CVLIs e prisões por período (mensal, semanal, horário).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import json
from collections import defaultdict
import re
import unicodedata

# Paths
DATA_RAW = Path("data/raw")
OUTPUTS = Path("outputs")
DOCS = OUTPUTS / "docs"
DOCS.mkdir(exist_ok=True)

MONTHLY_CSV = OUTPUTS / "sazonalidade_bairro_cidade_monthly.csv"
WEEKDAY_CSV = OUTPUTS / "sazonalidade_bairro_cidade_weekday.csv"
HOURLY_CSV = OUTPUTS / "sazonalidade_bairro_cidade_hourly.csv"

PRISOES_FILE = DATA_RAW / "ocorrencia_policial_operacional.json"


def normalizar_texto(texto):
    """Normaliza texto: maiúsculas, sem acentos, trim."""
    if not texto or pd.isna(texto):
        return ""
    texto = str(texto).upper().strip()
    # Remove acentos
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto)
                    if unicodedata.category(c) != 'Mn')
    return texto


def load_data():
    """Carrega dados de CVLI (sazonalidade) e prisões operacionais."""
    print("Carregando dados...")
    
    if not MONTHLY_CSV.exists():
        print(f"Arquivo {MONTHLY_CSV} não encontrado")
        return None, None
    
    monthly = pd.read_csv(MONTHLY_CSV)
    weekday = pd.read_csv(WEEKDAY_CSV)
    hourly = pd.read_csv(HOURLY_CSV)
    
    if not PRISOES_FILE.exists():
        print(f"Arquivo {PRISOES_FILE} não encontrado")
        return (monthly, weekday, hourly), None
    
    try:
        with open(PRISOES_FILE, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except Exception as e:
        print(f"Erro ao carregar prisões: {e}")
        return (monthly, weekday, hourly), None
    
    # Extrair registros (esperado: lista com wrapper SQL)
    records = []
    if isinstance(raw_data, list):
        # Procurar por item do tipo 'table' com 'data'
        for item in raw_data:
            if isinstance(item, dict) and item.get('type') == 'table' and 'data' in item:
                records = item['data']
                break
    elif isinstance(raw_data, dict):
        if 'data' in raw_data:
            records = raw_data.get('data', [])
        else:
            records = [raw_data]
    else:
        records = []
    
    print(f"CVLI mensal: {monthly.shape}")
    print(f"Registros prisões (raw): {len(records)}")
    print(f"\nColunas CVLI: {monthly.columns.tolist()}")
    
    # Converter para DataFrame
    prisoes = pd.DataFrame(records)
    print(f"Prisões DataFrame: {prisoes.shape}")
    print(f"Colunas Prisões: {prisoes.columns.tolist()}")
    
    return (monthly, weekday, hourly), prisoes


def compute_prisoes_sazonalidade(prisoes_df):
    """
    Computa sazonalidade de prisões (mensal, semanal, horário)
    a partir do arquivo ocorrencia_policial_operacional.json.
    """
    print("\n[1] Computando sazonalidade de Prisões...")
    
    prisoes_df = prisoes_df.copy()
    
    # Detectar coluna de data (Data, date, etc.)
    data_col = None
    for col in ['Data', 'data', 'date', 'DATE']:
        if col in prisoes_df.columns:
            data_col = col
            break
    
    if data_col is None:
        print("Coluna de data não encontrada")
        return None, None, None
    
    # Detectar coluna de bairro
    bairro_col = None
    for col in ['BairroOcor', 'bairro', 'Bairro', 'bairro_ocor', 'localidade']:
        if col in prisoes_df.columns:
            bairro_col = col
            break
    
    # Detectar coluna de cidade
    cidade_col = None
    for col in ['CidadeOcor', 'cidade', 'Cidade', 'municipio', 'municipio_ocor']:
        if col in prisoes_df.columns:
            cidade_col = col
            break
    
    print(f"  Data col: {data_col}, Bairro col: {bairro_col}, Cidade col: {cidade_col}")
    
    # Converter data para datetime
    prisoes_df[data_col] = pd.to_datetime(prisoes_df[data_col], errors='coerce')
    
    # Extrair tempo
    prisoes_df['mes'] = prisoes_df[data_col].dt.month
    prisoes_df['dia_semana'] = prisoes_df[data_col].dt.day_name()
    prisoes_df['hora'] = prisoes_df[data_col].dt.hour
    
    # Normalizar bairro e cidade
    if bairro_col:
        prisoes_df['bairro_norm'] = prisoes_df[bairro_col].apply(normalizar_texto)
    else:
        prisoes_df['bairro_norm'] = "DESCONHECIDO"
    
    if cidade_col:
        prisoes_df['cidade_norm'] = prisoes_df[cidade_col].apply(normalizar_texto)
    else:
        prisoes_df['cidade_norm'] = "DESCONHECIDO"
    
    # Remover NaT
    prisoes_df = prisoes_df.dropna(subset=[data_col])
    
    # Sazonalidade mensal (por bairro/cidade)
    monthly_pris = prisoes_df.groupby(['cidade_norm', 'mes']).size().reset_index(name='count')
    monthly_pris.columns = ['cidade', 'month', 'count']
    
    # Sazonalidade semanal
    weekday_pris = prisoes_df.groupby(['cidade_norm', 'dia_semana']).size().reset_index(name='count')
    weekday_pris.columns = ['cidade', 'weekday', 'count']
    
    # Sazonalidade horária
    hourly_pris = prisoes_df.groupby(['cidade_norm', 'hora']).size().reset_index(name='count')
    hourly_pris.columns = ['cidade', 'hour', 'count']
    
    print(f"✓ Prisões mensais: {monthly_pris.shape}")
    print(f"✓ Prisões semanais: {weekday_pris.shape}")
    print(f"✓ Prisões horárias: {hourly_pris.shape}")
    
    return monthly_pris, weekday_pris, hourly_pris


def analyze_correlation_by_city(cvli_monthly, prisoes_monthly):
    """Calcula correlação CVLI vs Prisões por cidade."""
    print("\n[2] Analisando correlações por cidade...")
    
    correlations = []
    
    if prisoes_monthly is None or cvli_monthly is None:
        return correlations
    
    # Normalizar CVLI
    cvli_monthly_norm = cvli_monthly.copy()
    cvli_monthly_norm['cidade_norm'] = cvli_monthly_norm['cidade'].apply(normalizar_texto)
    
    # Normalizar prisões (já devem estar normalizadas)
    prisoes_monthly_norm = prisoes_monthly.copy()
    prisoes_monthly_norm['cidade_norm'] = prisoes_monthly_norm['cidade'].apply(normalizar_texto)
    
    # Cruzar por cidade e período
    merged = pd.merge(
        cvli_monthly_norm[['cidade_norm', 'month', 'count']].rename(columns={'count': 'cvli_count'}),
        prisoes_monthly_norm[['cidade_norm', 'month', 'count']].rename(columns={'count': 'pris_count'}),
        on=['cidade_norm', 'month'],
        how='inner'
    )
    
    print(f"Registros cruzados: {len(merged)}")
    
    for city in merged['cidade_norm'].unique():
        city_data = merged[merged['cidade_norm'] == city]
        
        if len(city_data) >= 3:  # Mínimo para correlação
            cvli_vals = city_data['cvli_count'].values
            pris_vals = city_data['pris_count'].values
            
            try:
                corr_pearson, p_value = pearsonr(cvli_vals, pris_vals)
                corr_spearman, p_spearman = spearmanr(cvli_vals, pris_vals)
                
                # Calcular tendência
                cvli_trend = (cvli_vals[-1] - cvli_vals[0]) / max(cvli_vals[0], 1)
                pris_trend = (pris_vals[-1] - pris_vals[0]) / max(pris_vals[0], 1)
                
                correlations.append({
                    'cidade': city,
                    'n_periodos': len(city_data),
                    'corr_pearson': round(corr_pearson, 3),
                    'corr_spearman': round(corr_spearman, 3),
                    'p_value': round(p_value, 4),
                    'cvli_trend': round(cvli_trend, 3),
                    'pris_trend': round(pris_trend, 3),
                    'cvli_media': round(cvli_vals.mean(), 2),
                    'pris_media': round(pris_vals.mean(), 2),
                })
            except Exception as e:
                pass
    
    return correlations


def find_divergence_patterns(cvli_monthly, prisoes_monthly):
    """
    Identifica padrões de divergência forte:
    - Quando CVLI aumenta mas prisões diminuem (ou vice-versa)
    - Disparidade significativa
    """
    print("\n[3] Buscando padrões de divergência (aumento/diminuição forte)...")
    
    patterns = []
    
    if prisoes_monthly is None or cvli_monthly is None:
        return patterns
    
    # Normalizar CVLI
    cvli_monthly_norm = cvli_monthly.copy()
    cvli_monthly_norm['cidade_norm'] = cvli_monthly_norm['cidade'].apply(normalizar_texto)
    
    # Normalizar prisões
    prisoes_monthly_norm = prisoes_monthly.copy()
    prisoes_monthly_norm['cidade_norm'] = prisoes_monthly_norm['cidade'].apply(normalizar_texto)
    
    # Agregar por cidade e mês
    cvli_agg = cvli_monthly_norm.groupby(['cidade_norm', 'month'])['count'].sum().reset_index(name='cvli_count')
    pris_agg = prisoes_monthly_norm.groupby(['cidade_norm', 'month'])['count'].sum().reset_index(name='pris_count')
    
    merged = pd.merge(cvli_agg, pris_agg, left_on=['cidade_norm', 'month'], right_on=['cidade_norm', 'month'], how='inner')
    
    for city in merged['cidade_norm'].unique():
        city_data = merged[merged['cidade_norm'] == city].sort_values('month')
        
        if len(city_data) >= 2:
            for i in range(len(city_data) - 1):
                row_curr = city_data.iloc[i]
                row_next = city_data.iloc[i + 1]
                
                cvli_change = row_next['cvli_count'] - row_curr['cvli_count']
                pris_change = row_next['pris_count'] - row_curr['pris_count']
                
                # Normalizar por valores anteriores
                cvli_pct_change = (cvli_change / max(row_curr['cvli_count'], 1)) * 100
                pris_pct_change = (pris_change / max(row_curr['pris_count'], 1)) * 100
                
                # Padrões fortes: mudanças em direções opostas e significativas
                if abs(cvli_pct_change) > 30 or abs(pris_pct_change) > 30:
                    # Divergência: uma sobe, outra desce
                    if (cvli_pct_change > 30 and pris_pct_change < -30) or \
                       (cvli_pct_change < -30 and pris_pct_change > 30):
                        patterns.append({
                            'cidade': city,
                            'mes_inicial': int(row_curr['month']),
                            'mes_final': int(row_next['month']),
                            'cvli_inicial': int(row_curr['cvli_count']),
                            'cvli_final': int(row_next['cvli_count']),
                            'cvli_pct_change': round(cvli_pct_change, 1),
                            'pris_inicial': int(row_curr['pris_count']),
                            'pris_final': int(row_next['pris_count']),
                            'pris_pct_change': round(pris_pct_change, 1),
                            'tipo_divergencia': 'CVLI↑ Pris↓' if cvli_pct_change > 30 else 'CVLI↓ Pris↑',
                        })
    
    return patterns


def generate_report(correlations, patterns):
    """Gera relatório MD com achados."""
    print("\n[4] Gerando relatório...")
    
    lines = []
    lines.append("# Análise Cruzada: Sazonalidade CVLI × Prisões")
    lines.append("")
    lines.append("## Objetivo")
    lines.append("Identificar correlações, divergências e padrões de aumento/diminuição")
    lines.append("entre CVLIs e prisões por período temporal (mensal, semanal, horário).")
    lines.append("")
    
    # Correlações
    lines.append("## 1. Correlações entre CVLI e Prisões (por Cidade)")
    lines.append("")
    
    if correlations:
        corr_df = pd.DataFrame(correlations).sort_values('corr_pearson', ascending=False)
        
        lines.append("### Cidades com CORRELAÇÃO FORTE positiva (CVLI ↔ Prisões)")
        lines.append("")
        
        strong_pos = corr_df[corr_df['corr_pearson'] > 0.7]
        if len(strong_pos) > 0:
            for _, row in strong_pos.head(10).iterrows():
                lines.append(f"- **{row['cidade']}**: corr={row['corr_pearson']:.3f}, "
                           f"p={row['p_value']}, CVLI_trend={row['cvli_trend']}, Pris_trend={row['pris_trend']}")
        else:
            lines.append("(Nenhuma correlação > 0.7)")
        
        lines.append("")
        lines.append("### Cidades com CORRELAÇÃO MODERADA (0.3 - 0.7)")
        lines.append("")
        
        moderate = corr_df[(corr_df['corr_pearson'] > 0.3) & (corr_df['corr_pearson'] <= 0.7)]
        if len(moderate) > 0:
            for _, row in moderate.head(10).iterrows():
                lines.append(f"- {row['cidade']}: corr={row['corr_pearson']:.3f}")
        else:
            lines.append("(Nenhuma)")
        
        lines.append("")
        lines.append("### Cidades com CORRELAÇÃO FRACA ou NEGATIVA (<0.3)")
        lines.append("")
        
        weak = corr_df[corr_df['corr_pearson'] <= 0.3]
        if len(weak) > 0:
            lines.append(f"Total de cidades com baixa correlação: {len(weak)}")
            for _, row in weak.head(5).iterrows():
                lines.append(f"- {row['cidade']}: corr={row['corr_pearson']:.3f}")
        
        # Salvar CSV
        csv_path = DOCS / "cvli_prisoes_correlacao_por_cidade.csv"
        corr_df.to_csv(csv_path, index=False)
        lines.append(f"\n**CSV Salvo:** {csv_path}")
    else:
        lines.append("(Sem dados disponíveis para correlação)")
    
    # Divergências
    lines.append("\n## 2. Padrões de Divergência Forte (Aumento/Diminuição Oposta)")
    lines.append("")
    
    if patterns:
        patterns_df = pd.DataFrame(patterns).sort_values('cvli_pct_change', key=abs, ascending=False)
        
        lines.append(f"### Encontrados {len(patterns)} padrões de divergência")
        lines.append("")
        
        for _, row in patterns_df.head(15).iterrows():
            lines.append(f"**{row['cidade']}** (Mês {row['mes_inicial']} → {row['mes_final']})")
            lines.append(f"- CVLI: {row['cvli_inicial']} → {row['cvli_final']} ({row['cvli_pct_change']:+.1f}%)")
            lines.append(f"- Prisões: {row['pris_inicial']} → {row['pris_final']} ({row['pris_pct_change']:+.1f}%)")
            lines.append(f"- Padrão: **{row['tipo_divergencia']}**")
            lines.append("")
        
        # Salvar CSV
        csv_path = DOCS / "cvli_prisoes_divergencias_forte.csv"
        patterns_df.to_csv(csv_path, index=False)
        lines.append(f"**CSV Salvo:** {csv_path}")
    else:
        lines.append("(Sem padrões significativos detectados)")
    
    lines.append("\n---")
    lines.append("**Análise gerada em:** 22 de janeiro de 2026")
    
    # Salvar MD
    md_path = DOCS / "ANALISE_CVLI_PRISOES_CRUZADO.md"
    md_path.write_text("\n".join(lines), encoding='utf-8')
    print(f"✓ Relatório MD salvo: {md_path}")
    
    return md_path


def main():
    print("="*60)
    print("Cruzamento: Sazonalidade CVLI × Prisões")
    print("="*60)
    
    # Carregar dados
    (cvli_monthly, cvli_weekday, cvli_hourly), prisoes = load_data()
    
    if cvli_monthly is None:
        print("Erro ao carregar dados de CVLI")
        return
    
    if prisoes is None:
        print("Erro ao carregar dados de prisões")
        return
    
    # Computar sazonalidade de prisões
    pris_monthly, pris_weekday, pris_hourly = compute_prisoes_sazonalidade(prisoes)
    
    if pris_monthly is None:
        print("Erro ao computar sazonalidade de prisões")
        return
    
    # Análises
    correlations = analyze_correlation_by_city(cvli_monthly, pris_monthly)
    patterns = find_divergence_patterns(cvli_monthly, pris_monthly)
    
    # Relatório
    generate_report(correlations, patterns)
    
    print("\n" + "="*60)
    print("✅ Análise cruzada concluída com sucesso!")
    print("="*60)


if __name__ == '__main__':
    main()
