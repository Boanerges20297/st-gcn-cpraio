#!/usr/bin/env python3
"""
ANÁLISE DE MOVIMENTAÇÃO DE FACÇÕES
Processa GeoJSONs de facções por data para criar features dinâmicas
de mudança de controle territorial
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from datetime import datetime, timedelta
import logging
from collections import defaultdict

# --- Setup de Paths e Imports ---
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

import config

# --- Configurar Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# STAGE 1: DESCOBRIR SNAPSHOTS DE FACÇÕES POR DATA
# ============================================================================

def find_faction_snapshots():
    """
    Encontra todas as pastas de facções organizadas por data
    Retorna dicionário: {date: path_to_folder}
    """
    logger.info("[STAGE 1] Descobrindo snapshots de facções por data...")
    
    snapshots = {}
    faction_base = config.DATA_GRAPH
    
    # Procurar pastas faccoes_DD_MM_YYYY
    import re
    pattern = re.compile(r'faccoes_(\d{2})_(\d{2})_(\d{4})')
    
    for item in faction_base.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                day, month, year = match.groups()
                date = pd.to_datetime(f"{year}-{month}-{day}")
                snapshots[date] = item
    
    # Ordenar por data
    snapshots = dict(sorted(snapshots.items()))
    
    logger.info(f"✓ {len(snapshots)} snapshots de facções encontrados:")
    for date, path in snapshots.items():
        geojsons = list(path.glob("*.geojson"))
        logger.info(f"  - {date.strftime('%d/%m/%Y')}: {len(geojsons)} facções")
    
    return snapshots

def load_faction_geojsons(snapshot_path):
    """
    Carrega todos os GeoJSONs de uma data específica
    Retorna dicionário: {faction_name: GeoDataFrame}
    """
    factions = {}
    
    for geojson_file in snapshot_path.glob("*.geojson"):
        faction_name = geojson_file.stem  # Nome sem extensão
        try:
            gdf = gpd.read_file(geojson_file)
            factions[faction_name] = gdf
        except Exception as e:
            logger.warning(f"Erro ao ler {geojson_file.name}: {e}")
    
    return factions

# ============================================================================
# STAGE 2: MAPEAR CONTROLE TERRITORIAL POR BAIRRO E DATA
# ============================================================================

def map_bairro_faction_timeseries(snapshots, metadata):
    """
    Cria série temporal de qual facção controla cada bairro
    
    Retorna:
    - bairro_faction_ts: dict {bairro: {date: [factions]}}
    - faction_changes: dict {bairro: [(date, old_faction, new_faction)]}
    """
    logger.info("[STAGE 2] Mapeando controle territorial por bairro e data...")
    
    bairro_list = metadata['bairro_lista']
    bairro_faction_ts = {bairro: {} for bairro in bairro_list}
    faction_changes = defaultdict(list)
    
    # Processar cada snapshot
    prev_snapshot = None
    prev_date = None
    
    for date, snapshot_path in sorted(snapshots.items()):
        logger.info(f"  → Processando {date.strftime('%d/%m/%Y')}...")
        
        factions = load_faction_geojsons(snapshot_path)
        
        # Para cada bairro, determinar qual facção o controla
        for bairro in bairro_list:
            controlling_factions = []
            
            for faction_name, gdf in factions.items():
                if gdf.empty:
                    continue
                
                # Verificar se bairro aparece em alguma geometria
                # (Simplificado: assume que nome do bairro está nas propriedades)
                if 'name' in gdf.columns:
                    matches = gdf[gdf['name'].str.contains(bairro, case=False, na=False)]
                    if not matches.empty:
                        controlling_factions.append(faction_name)
            
            # Se não encontrou por nome, assume "sem_controle"
            if not controlling_factions:
                controlling_factions = ['SEM_CONTROLE']
            
            bairro_faction_ts[bairro][date] = controlling_factions
            
            # Detectar mudanças
            if prev_date is not None and bairro in bairro_faction_ts:
                old_factions = bairro_faction_ts[bairro].get(prev_date, ['SEM_CONTROLE'])
                new_factions = controlling_factions
                
                # Se mudou de facção
                if set(old_factions) != set(new_factions):
                    for new_fac in new_factions:
                        if new_fac not in old_factions:
                            faction_changes[bairro].append((date, old_factions[0], new_fac))
        
        prev_snapshot = factions
        prev_date = date
    
    logger.info(f"✓ Mapeamento completo: {len(bairro_faction_ts)} bairros")
    logger.info(f"✓ Detectadas {sum(len(v) for v in faction_changes.values())} mudanças de controle")
    
    return bairro_faction_ts, faction_changes

# ============================================================================
# STAGE 3: CRIAR FEATURES DE DINÂMICA TERRITORIAL
# ============================================================================

def create_faction_features(metadata, bairro_faction_ts, faction_changes):
    """
    Cria matriz de features de dinâmica de facções
    
    Features por bairro-dia:
    1. faction_change_indicator: 1 se houve mudança de controle naquele dia
    2. faction_stability: dias desde última mudança (0-max_days)
    3. faction_conflict_risk: número de facções disputando o mesmo bairro
    4. territorial_volatility: mudanças nos últimos 30 dias
    """
    logger.info("[STAGE 3] Criando features de dinâmica territorial...")
    
    num_days = metadata['num_dias']
    num_bairros = metadata['num_bairros']
    
    periodo_inicio = pd.to_datetime(metadata['periodo_inicio'])
    all_dates = pd.date_range(periodo_inicio, periods=num_days, freq='D')
    
    # Inicializar tensores de features
    features_tensor = np.zeros((num_days, num_bairros, 4), dtype=np.float32)
    # Dimensões: [dias, bairros, 4 features]
    # - 0: change_indicator (0 ou 1)
    # - 1: stability (dias desde última mudança)
    # - 2: conflict_risk (0-1)
    # - 3: volatility (mudanças nos últimos 30 dias)
    
    bairro_to_idx = {bairro: i for i, bairro in enumerate(metadata['bairro_lista'])}
    
    for bairro_idx, bairro in enumerate(metadata['bairro_lista']):
        if bairro not in bairro_faction_ts:
            continue
        
        faction_timeline = bairro_faction_ts[bairro]
        changes = {date: True for date, _, _ in faction_changes[bairro]}
        
        last_change_day = -np.inf
        
        for day_idx, current_date in enumerate(all_dates):
            # Feature 0: Mudança de controle neste dia
            if current_date in changes:
                features_tensor[day_idx, bairro_idx, 0] = 1.0
                last_change_day = day_idx
            
            # Feature 1: Estabilidade (dias desde última mudança)
            stability = day_idx - last_change_day
            features_tensor[day_idx, bairro_idx, 1] = min(stability, 365)  # Cap em 365 dias
            
            # Feature 2: Risco de conflito (múltiplas facções)
            current_factions = faction_timeline.get(current_date, ['SEM_CONTROLE'])
            num_factions = len([f for f in current_factions if f != 'SEM_CONTROLE'])
            conflict_risk = min(num_factions / 3.0, 1.0)  # Normalizar para 0-1
            features_tensor[day_idx, bairro_idx, 2] = conflict_risk
            
            # Feature 3: Volatilidade nos últimos 30 dias
            window_start = max(0, day_idx - 30)
            changes_in_window = np.sum(features_tensor[window_start:day_idx, bairro_idx, 0])
            volatility = min(changes_in_window / 30.0, 1.0)  # Normalizar
            features_tensor[day_idx, bairro_idx, 3] = volatility
    
    logger.info(f"✓ Tensor de features de dinâmica criado: {features_tensor.shape}")
    logger.info(f"  - Mudanças detectadas: {np.sum(features_tensor[:, :, 0]):.0f}")
    logger.info(f"  - Estabilidade média: {np.mean(features_tensor[:, :, 1]):.1f} dias")
    logger.info(f"  - Risco de conflito médio: {np.mean(features_tensor[:, :, 2]):.3f}")
    logger.info(f"  - Volatilidade média: {np.mean(features_tensor[:, :, 3]):.3f}")
    
    return features_tensor

# ============================================================================
# STAGE 4: INTEGRAR COM TENSOR EXISTENTE
# ============================================================================

def merge_faction_features_to_tensor(tensor_multivariado, faction_features_tensor, metadata):
    """
    Combina tensor multivariado (CVLI, Prisões, Apreensões)
    com features de dinâmica de facções
    
    Novo shape: (num_days, num_bairros, 7)
    - 0-2: CVLI, Prisões, Apreensões (original)
    - 3: Change indicator
    - 4: Stability
    - 5: Conflict risk
    - 6: Volatility
    """
    logger.info("[STAGE 4] Integrando features de facções ao tensor...")
    
    # Validar shapes
    assert tensor_multivariado.shape[:2] == faction_features_tensor.shape[:2], \
        f"Shape mismatch: {tensor_multivariado.shape} vs {faction_features_tensor.shape}"
    
    # Concatenar ao longo da dimensão de features
    enhanced_tensor = np.concatenate(
        [tensor_multivariado, faction_features_tensor],
        axis=2
    )
    
    logger.info(f"✓ Tensor multivariado + facções criado: {enhanced_tensor.shape}")
    logger.info(f"  - Features por bairro-dia: {enhanced_tensor.shape[2]}")
    logger.info(f"    * 0-2: CVLI, Prisões, Apreensões")
    logger.info(f"    * 3: Mudança de controle territorial")
    logger.info(f"    * 4: Estabilidade (dias)")
    logger.info(f"    * 5: Risco de conflito")
    logger.info(f"    * 6: Volatilidade territorial")
    
    return enhanced_tensor

# ============================================================================
# STAGE 5: SALVAMENTO E ANÁLISE
# ============================================================================

def save_faction_analysis(bairro_faction_ts, faction_changes, snapshots, metadata):
    """Salva análise de movimentação de facções"""
    logger.info("[STAGE 5] Salvando análise de facções...")
    
    # Sumário de movimentação por bairro
    summary = {}
    for bairro in metadata['bairro_lista']:
        if bairro in bairro_faction_ts:
            changes_count = len(faction_changes[bairro])
            factions_list = set()
            for factions in bairro_faction_ts[bairro].values():
                factions_list.update(factions)
            
            summary[bairro] = {
                'mudancas_totais': changes_count,
                'facoes_envolvidas': list(factions_list),
            }
    
    # Salvar JSON
    output_path = config.DATA_PROCESSED / 'analise_movimentacao_faccoes.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Análise salva em: {output_path}")
    
    # Criar dataframe de mudanças para inspeção
    changes_list = []
    for bairro, bairro_changes in faction_changes.items():
        for date, old_fac, new_fac in bairro_changes:
            changes_list.append({
                'data': date,
                'bairro': bairro,
                'facacao_anterior': old_fac,
                'facacao_nova': new_fac,
            })
    
    if changes_list:
        df_changes = pd.DataFrame(changes_list)
        df_changes.to_csv(config.DATA_PROCESSED / 'historico_mudancas_territoriais.csv', index=False)
        logger.info(f"✓ Histórico de mudanças salvo: {len(df_changes)} eventos")

def save_enhanced_tensor(enhanced_tensor, metadata):
    """Salva tensor enriquecido com features de facções"""
    logger.info("[STAGE 5] Salvando tensor enriquecido...")
    
    output_path = config.DATA_PROCESSED / 'tensor_cvli_prisoes_faccoes.npy'
    np.save(output_path, enhanced_tensor.astype(np.float32))
    
    logger.info(f"✓ Tensor salvo em: {output_path}")
    logger.info(f"  - Shape: {enhanced_tensor.shape}")
    logger.info(f"  - Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Atualizar metadata
    metadata_path = config.DATA_PROCESSED / 'metadata_producao_v2.json'
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata_dict = json.load(f)
    
    metadata_dict['tensor_enriquecido_shape'] = list(enhanced_tensor.shape)
    metadata_dict['tensor_enriquecido_features'] = [
        'CVLI', 'Prisoes', 'Apreensoes',
        'Mudanca_Territorial', 'Estabilidade', 'Risco_Conflito', 'Volatilidade'
    ]
    metadata_dict['versao'] = 'v2_com_dinamica_faccoes'
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Metadata atualizado")

def generate_faction_report(snapshots, bairro_faction_ts, faction_changes, metadata):
    """Gera relatório detalhado de dinâmica de facções"""
    logger.info("[STAGE 5] Gerando relatório de facções...")
    
    report = []
    report.append("# ANÁLISE DE MOVIMENTAÇÃO DE FACÇÕES\n")
    report.append(f"**Gerado em:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
    report.append(f"**Período:** {metadata['periodo_inicio']} a {metadata['periodo_fim']}\n\n")
    
    # Sumário geral
    total_changes = sum(len(v) for v in faction_changes.values())
    report.append(f"## Sumário\n")
    report.append(f"- **Snapshots de facções:** {len(snapshots)}\n")
    report.append(f"- **Total de mudanças territoriais:** {total_changes}\n")
    report.append(f"- **Bairros com mudanças:** {sum(1 for v in faction_changes.values() if v)}\n\n")
    
    # Top bairros com mais mudanças
    report.append(f"## Bairros com Maior Volatilidade Territorial\n\n")
    bairro_volatility = [(b, len(c)) for b, c in faction_changes.items() if c]
    bairro_volatility.sort(key=lambda x: x[1], reverse=True)
    
    for i, (bairro, changes_count) in enumerate(bairro_volatility[:15], 1):
        report.append(f"{i}. **{bairro}**: {changes_count} mudanças\n")
    
    report.append("\n## Cronologia de Mudanças\n\n")
    
    # Agrupar por data
    changes_by_date = defaultdict(list)
    for bairro, bairro_changes in faction_changes.items():
        for date, old_fac, new_fac in bairro_changes:
            changes_by_date[date].append((bairro, old_fac, new_fac))
    
    for date in sorted(changes_by_date.keys()):
        changes = changes_by_date[date]
        report.append(f"### {date.strftime('%d/%m/%Y')}\n\n")
        for bairro, old_fac, new_fac in changes:
            report.append(f"- **{bairro}**: {old_fac} → {new_fac}\n")
        report.append("\n")
    
    # Salvar
    report_path = config.DATA_PROCESSED / 'RELATORIO_DINAMICA_FACCOES.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report)
    
    logger.info(f"✓ Relatório salvo em: {report_path}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("ANÁLISE DE MOVIMENTAÇÃO DE FACÇÕES - INTEGRAÇÃO AO MODELO")
    print("="*80)
    print()
    
    # Stage 1: Descobrir snapshots
    snapshots = find_faction_snapshots()
    if not snapshots:
        logger.error("❌ Nenhum snapshot de facções encontrado!")
        return False
    
    # Carregar metadata
    with open(config.DATA_PROCESSED / 'metadata_producao_v2.json', 'r') as f:
        raw_metadata = json.load(f)
    
    metadata = {
        'periodo_inicio': raw_metadata['periodo'].split(' a ')[0],
        'periodo_fim': raw_metadata['periodo'].split(' a ')[1],
        'num_dias': raw_metadata['dias'],
        'num_bairros': raw_metadata['bairros'],
        'bairro_lista': raw_metadata['bairros_normalizados'],
    }
    
    # Stage 2: Mapear controle territorial
    bairro_faction_ts, faction_changes = map_bairro_faction_timeseries(snapshots, metadata)
    
    # Stage 3: Criar features
    faction_features_tensor = create_faction_features(metadata, bairro_faction_ts, faction_changes)
    
    # Stage 4: Integrar com tensor existente
    tensor_multivariado = np.load(config.DATA_PROCESSED / 'tensor_multivariado.npy')
    enhanced_tensor = merge_faction_features_to_tensor(tensor_multivariado, faction_features_tensor, metadata)
    
    # Stage 5: Salvar e gerar relatórios
    save_faction_analysis(bairro_faction_ts, faction_changes, snapshots, metadata)
    save_enhanced_tensor(enhanced_tensor, metadata)
    generate_faction_report(snapshots, bairro_faction_ts, faction_changes, metadata)
    
    print("\n" + "="*80)
    print("✅ ANÁLISE DE FACÇÕES CONCLUÍDA COM SUCESSO")
    print("="*80)
    print(f"\nTensor enriquecido: data/processed/tensor_cvli_prisoes_faccoes.npy")
    print(f"Shape: {enhanced_tensor.shape}")
    print(f"\nFeatures (7 dimensões por bairro-dia):")
    print(f"  0. CVLI")
    print(f"  1. Prisões")
    print(f"  2. Apreensões")
    print(f"  3. 🚨 Mudança de controle territorial")
    print(f"  4. 📊 Estabilidade (dias desde última mudança)")
    print(f"  5. ⚔️  Risco de conflito (0-1)")
    print(f"  6. 🌊 Volatilidade territorial (mudanças/30 dias)")
    print(f"\nRelatórios gerados:")
    print(f"  - RELATORIO_DINAMICA_FACCOES.md (cronologia detalhada)")
    print(f"  - analise_movimentacao_faccoes.json (sumário por bairro)")
    print(f"  - historico_mudancas_territoriais.csv (timeline)")
    print()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
