"""
Análise Avançada: Impacto de Prisões sobre CVLI
Identifica:
1. Cidades onde aumento de prisões → diminuição de CVLI (correlação negativa forte = sucesso operacional)
2. Cidades onde aumento de prisões → aumento de CVLI (sem efeito ou contraproducente)
3. Bairros com "efetividade de prisões" (redução de crimes após operações)
4. Padrões sazonais: meses com mais prisões x menos crimes
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from scipy.stats import pearsonr
import unicodedata

# Paths
OUTPUTS = Path("outputs")
DOCS = OUTPUTS / "docs"
DOCS.mkdir(exist_ok=True)

MONTHLY_CSV = OUTPUTS / "sazonalidade_bairro_cidade_monthly.csv"
PRISOES_CSV = OUTPUTS / "sazonalidade_bairro_cidade_monthly.csv"  # Será criado no script anterior

DATA_RAW = Path("data/raw")
PRISOES_FILE = DATA_RAW / "ocorrencia_policial_operacional.json"


def normalizar_texto(texto):
    """Normaliza texto: maiúsculas, sem acentos, trim."""
    if not texto or pd.isna(texto):
        return ""
    texto = str(texto).upper().strip()
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto)
                    if unicodedata.category(c) != 'Mn')
    return texto


def load_all_data():
    """Carrega dados de CVLI e prisões."""
    print("[LOAD] Carregando dados...")
    
    # CVLI
    cvli = pd.read_csv(MONTHLY_CSV)
    cvli['cidade_norm'] = cvli['cidade'].apply(normalizar_texto)
    
    # Prisões (carregar e processar)
    with open(PRISOES_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    records = []
    for item in raw_data:
        if isinstance(item, dict) and item.get('type') == 'table' and 'data' in item:
            records = item['data']
            break
    
    prisoes_df = pd.DataFrame(records)
    prisoes_df['Data'] = pd.to_datetime(prisoes_df['Data'], errors='coerce')
    prisoes_df['mes'] = prisoes_df['Data'].dt.month
    prisoes_df['ano'] = prisoes_df['Data'].dt.year
    prisoes_df['cidade_norm'] = prisoes_df['CidadeOcor'].apply(normalizar_texto)
    
    # Contar prisões por cidade/mês
    prisoes_agg = prisoes_df.groupby(['cidade_norm', 'mes']).size().reset_index(name='prisoes_count')
    
    return cvli, prisoes_agg


def analyze_effectiveness(cvli, prisoes):
    """
    Analisa efetividade: cidades onde mais prisões = menos crimes
    """
    print("\n[1] Analisando EFETIVIDADE de Prisões (mais prisões → menos crimes)...")
    
    # Cruzar por cidade e mês
    merged = pd.merge(
        cvli[['cidade_norm', 'month', 'count']].rename(columns={'count': 'cvli_count', 'month': 'mes'}),
        prisoes[['cidade_norm', 'mes', 'prisoes_count']],
        on=['cidade_norm', 'mes'],
        how='inner'
    )
    
    print(f"Registros cruzados: {len(merged)}")
    
    effectiveness = []
    for city in merged['cidade_norm'].unique():
        city_data = merged[merged['cidade_norm'] == city]
        
        if len(city_data) >= 4:  # Mínimo para análise
            cvli_vals = city_data['cvli_count'].values.astype(float)
            pris_vals = city_data['prisoes_count'].values.astype(float)
            
            try:
                # Correlação entre prisões e CVLI
                corr, p_val = pearsonr(pris_vals, cvli_vals)
                
                # Efetividade: CORRELAÇÃO NEGATIVA = sucesso operacional
                # (mais prisões, menos crimes)
                
                # Também calcular: redução média de CVLI para cada prisão
                if pris_vals.sum() > 0:
                    crime_reduction_ratio = -cvli_vals.sum() / pris_vals.sum()  # Ideal: negativo grande
                else:
                    crime_reduction_ratio = 0
                
                # Categorizar efetividade
                if corr < -0.5:
                    efetividade = "MUITO ALTA (↓↓)"
                elif corr < -0.2:
                    efetividade = "ALTA (↓)"
                elif corr > 0.5:
                    efetividade = "INEFICAZ (↑↑)"
                elif corr > 0.2:
                    efetividade = "BAIXA (↑)"
                else:
                    efetividade = "NEUTRA (=)"
                
                effectiveness.append({
                    'cidade': city,
                    'n_periodos': len(city_data),
                    'corr_prisoes_cvli': round(corr, 3),
                    'p_value': round(p_val, 4),
                    'crime_reduction_ratio': round(crime_reduction_ratio, 3),
                    'total_cvli': int(cvli_vals.sum()),
                    'total_prisoes': int(pris_vals.sum()),
                    'efetividade_categoria': efetividade,
                    'cvli_media': round(cvli_vals.mean(), 2),
                    'prisoes_media': round(pris_vals.mean(), 2),
                })
            except Exception as e:
                pass
    
    effectiveness_df = pd.DataFrame(effectiveness).sort_values('corr_prisoes_cvli')
    
    print(f"✓ {len(effectiveness_df)} cidades analisadas")
    
    return effectiveness_df


def find_impact_patterns(cvli, prisoes):
    """
    Identifica períodos com impacto forte:
    - Mês com muitas prisões seguido de queda em CVLI
    - Mês com poucas prisões seguido de pico em CVLI
    """
    print("\n[2] Buscando PADRÕES DE IMPACTO (alta atividade operacional → resultado)...")
    
    # Cruzar por cidade e mês
    merged = pd.merge(
        cvli[['cidade_norm', 'month', 'count']].rename(columns={'count': 'cvli_count', 'month': 'mes'}),
        prisoes[['cidade_norm', 'mes', 'prisoes_count']],
        on=['cidade_norm', 'mes'],
        how='inner'
    )
    
    patterns = []
    for city in merged['cidade_norm'].unique():
        city_data = merged[merged['cidade_norm'] == city].sort_values('mes')
        
        if len(city_data) >= 2:
            for i in range(len(city_data) - 1):
                row_curr = city_data.iloc[i]
                row_next = city_data.iloc[i + 1]
                
                # Critério 1: Aumento de prisões (operação forte)
                pris_increase = row_next['prisoes_count'] - row_curr['prisoes_count']
                
                # Critério 2: Redução de CVLI (resultado esperado)
                cvli_change = row_next['cvli_count'] - row_curr['cvli_count']
                
                # Padrão positivo: pris ↑ E cvli ↓
                if pris_increase > 0 and cvli_change < 0:
                    cvli_reduction_pct = (cvli_change / max(row_curr['cvli_count'], 1)) * 100
                    pris_increase_pct = (pris_increase / max(row_curr['prisoes_count'], 1)) * 100
                    
                    patterns.append({
                        'cidade': city,
                        'mes_operacao': int(row_curr['mes']),
                        'mes_resultado': int(row_next['mes']),
                        'prisoes_antes': int(row_curr['prisoes_count']),
                        'prisoes_depois': int(row_next['prisoes_count']),
                        'prisoes_aumento': int(pris_increase),
                        'prisoes_aumento_pct': round(pris_increase_pct, 1),
                        'cvli_antes': int(row_curr['cvli_count']),
                        'cvli_depois': int(row_next['cvli_count']),
                        'cvli_reducao': int(abs(cvli_change)),
                        'cvli_reducao_pct': round(cvli_reduction_pct, 1),
                        'tipo_impacto': 'POSITIVO: Pris↑ Cvli↓',
                    })
                
                # Padrão negativo: pris ↑ MAS cvli ↑ (operação sem efeito)
                elif pris_increase > 0 and cvli_change > 0:
                    cvli_increase_pct = (cvli_change / max(row_curr['cvli_count'], 1)) * 100
                    
                    if cvli_increase_pct > 30:  # Só incluir aumentos significativos
                        patterns.append({
                            'cidade': city,
                            'mes_operacao': int(row_curr['mes']),
                            'mes_resultado': int(row_next['mes']),
                            'prisoes_antes': int(row_curr['prisoes_count']),
                            'prisoes_depois': int(row_next['prisoes_count']),
                            'prisoes_aumento': int(pris_increase),
                            'prisoes_aumento_pct': round((pris_increase / max(row_curr['prisoes_count'], 1)) * 100, 1),
                            'cvli_antes': int(row_curr['cvli_count']),
                            'cvli_depois': int(row_next['cvli_count']),
                            'cvli_aumento': int(cvli_change),
                            'cvli_aumento_pct': round(cvli_increase_pct, 1),
                            'tipo_impacto': 'NEGATIVO: Pris↑ Cvli↑',
                        })
    
    patterns_df = pd.DataFrame(patterns)
    print(f"✓ {len(patterns_df)} padrões de impacto encontrados")
    
    return patterns_df


def generate_advanced_report(effectiveness_df, patterns_df):
    """Gera relatório MD completo com análises."""
    print("\n[3] Gerando relatório avançado...")
    
    lines = []
    lines.append("# Análise de Impacto: Prisões vs CVLI")
    lines.append("")
    lines.append("**Objetivo**: Quantificar efetividade de operações policiais")
    lines.append("sobre redução/controle de Crimes Violentos Letais Intencionais")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Seção 1: Efetividade geral
    lines.append("## 1. Efetividade Geral por Cidade")
    lines.append("")
    
    if len(effectiveness_df) > 0:
        # Cidades com MUITO ALTA efetividade (correlação negativa < -0.5)
        very_high = effectiveness_df[effectiveness_df['corr_prisoes_cvli'] < -0.5]
        high = effectiveness_df[(effectiveness_df['corr_prisoes_cvli'] >= -0.5) & (effectiveness_df['corr_prisoes_cvli'] < -0.2)]
        neutral = effectiveness_df[(effectiveness_df['corr_prisoes_cvli'] >= -0.2) & (effectiveness_df['corr_prisoes_cvli'] <= 0.2)]
        low = effectiveness_df[(effectiveness_df['corr_prisoes_cvli'] > 0.2) & (effectiveness_df['corr_prisoes_cvli'] <= 0.5)]
        very_low = effectiveness_df[effectiveness_df['corr_prisoes_cvli'] > 0.5]
        
        lines.append("### 🟢 MUITO ALTA EFETIVIDADE (Prisões ↓↓ CVLI)")
        lines.append("")
        if len(very_high) > 0:
            lines.append("*(Mais prisões resulta em substancial redução de crimes)*")
            lines.append("")
            for _, row in very_high.head(10).iterrows():
                lines.append(f"- **{row['cidade']}**: corr={row['corr_prisoes_cvli']:.3f}, "
                           f"Total: {row['total_prisoes']} prisões → {row['total_cvli']} CVLI ({row['n_periodos']} períodos)")
        else:
            lines.append("*(Nenhuma cidade com correlação < -0.5)*")
        
        lines.append("")
        lines.append("### 🟡 ALTA EFETIVIDADE (Prisões ↓ CVLI)")
        lines.append("")
        if len(high) > 0:
            lines.append("*(Padrão claro: aumento de operações → queda de crimes)*")
            lines.append("")
            for _, row in high.head(10).iterrows():
                lines.append(f"- {row['cidade']}: corr={row['corr_prisoes_cvli']:.3f}, "
                           f"{row['total_prisoes']} prisões, {row['total_cvli']} CVLI")
        else:
            lines.append("*(Nenhuma cidade com correlação entre -0.5 e -0.2)*")
        
        lines.append("")
        lines.append("### ⚪ NEUTRA (sem padrão claro)")
        lines.append("")
        lines.append(f"**{len(neutral)} cidades** com correlação entre -0.2 e 0.2")
        
        lines.append("")
        lines.append("### 🔴 INEFICAZ (Prisões ↑↑ CVLI - SEM EFEITO ou PIORADO)")
        lines.append("")
        if len(very_low) > 0:
            lines.append("*(ALERTA: Aumento de operações NÃO reduz crimes - possível retalho, reorganização ou falta de integração)*")
            lines.append("")
            for _, row in very_low.head(10).iterrows():
                lines.append(f"- **{row['cidade']}**: corr={row['corr_prisoes_cvli']:.3f}, "
                           f"{row['total_prisoes']} prisões vs {row['total_cvli']} CVLI (↑↑↑)")
        else:
            lines.append("*(Nenhuma cidade com correlação > 0.5)*")
        
        # CSV de efetividade
        csv_path = DOCS / "efetividade_prisoes_por_cidade.csv"
        effectiveness_df.to_csv(csv_path, index=False)
        lines.append(f"\n**CSV**: {csv_path}")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Seção 2: Padrões de impacto
    lines.append("## 2. Padrões de Impacto Detectados")
    lines.append("")
    
    if len(patterns_df) > 0:
        positivos = patterns_df[patterns_df['tipo_impacto'].str.contains('POSITIVO')]
        negativos = patterns_df[patterns_df['tipo_impacto'].str.contains('NEGATIVO')]
        
        lines.append(f"### Operações com Resultado Positivo: {len(positivos)} casos")
        lines.append("")
        
        if len(positivos) > 0:
            lines.append("*(Período: aumento de prisões → queda subsequente de CVLI)*")
            lines.append("")
            
            positivos_sorted = positivos.sort_values('cvli_reducao_pct', ascending=False)
            for _, row in positivos_sorted.head(15).iterrows():
                lines.append(f"**{row['cidade']}** (Mês {row['mes_operacao']} → {row['mes_resultado']})")
                lines.append(f"- Prisões: {row['prisoes_antes']} → {row['prisoes_depois']} (+{row['prisoes_aumento']})")
                lines.append(f"- CVLI: {row['cvli_antes']} → {row['cvli_depois']} (↓{row['cvli_reducao_pct']:.1f}%)")
                lines.append("")
        
        lines.append("")
        lines.append(f"### Operações SEM Efeito (ou Contraproducentes): {len(negativos)} casos")
        lines.append("")
        
        if len(negativos) > 0:
            lines.append("*(ALERTA: Aumento de prisões mas CVLI também aumentou)*")
            lines.append("")
            
            negativos_sorted = negativos.sort_values('cvli_aumento_pct', ascending=False)
            for _, row in negativos_sorted.head(10).iterrows():
                lines.append(f"**{row['cidade']}** (Mês {row['mes_operacao']} → {row['mes_resultado']})")
                lines.append(f"- Prisões: {row['prisoes_antes']} → {row['prisoes_depois']} (+{row['prisoes_aumento']})")
                lines.append(f"- CVLI: {row['cvli_antes']} → {row['cvli_depois']} (↑{row['cvli_aumento_pct']:.1f}%) ⚠️")
                lines.append("")
        
        # CSV de padrões
        csv_path = DOCS / "impacto_prisoes_padroes.csv"
        patterns_df.to_csv(csv_path, index=False)
        lines.append(f"**CSV**: {csv_path}")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Seção 3: Recomendações
    lines.append("## 3. Recomendações Operacionais")
    lines.append("")
    lines.append("1. **Cidades com ALTA efetividade**: Manter/expandir operações RAIO (estratégia funcionando)")
    lines.append("2. **Cidades com BAIXA efetividade**: Revisar tática operacional (possível retalho, desorganização)")
    lines.append("3. **Cidades com padrão NEUTRO**: Integrar com outras inteligências (drogas, inteligência, fações)")
    lines.append("4. **Correlações NEGATIVAS (pior caso)**: Investigar possível aumento de retaliatória/conflitos")
    
    lines.append("")
    lines.append("---")
    lines.append(f"**Análise gerada em:** 22 de janeiro de 2026")
    
    # Salvar MD
    md_path = DOCS / "ANALISE_IMPACTO_PRISOES_AVANCADA.md"
    md_path.write_text("\n".join(lines), encoding='utf-8')
    print(f"✓ Relatório MD salvo: {md_path}")
    
    return md_path


def main():
    print("="*70)
    print("Análise Avançada: Impacto de Prisões sobre CVLI")
    print("="*70)
    
    # Carregar dados
    cvli, prisoes = load_all_data()
    
    print(f"  CVLI: {cvli.shape[0]} registros")
    print(f"  Prisões: {prisoes.shape[0]} registros")
    
    # Análises
    effectiveness_df = analyze_effectiveness(cvli, prisoes)
    patterns_df = find_impact_patterns(cvli, prisoes)
    
    # Relatório
    generate_advanced_report(effectiveness_df, patterns_df)
    
    print("\n" + "="*70)
    print("✅ Análise de impacto concluída com sucesso!")
    print("="*70)


if __name__ == '__main__':
    main()
