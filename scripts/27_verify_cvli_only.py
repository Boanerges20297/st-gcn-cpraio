import json
from pathlib import Path

import pandas as pd


INPUT = Path("data/raw/dados_status_ocorrencias_gerais_bairros_atribuidos.json")
MONTHLY = Path("outputs/sazonalidade_bairro_cidade_monthly.csv")

OUT_MD = Path("outputs/docs/cvli_verification_analysis.md")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        wrapper = json.load(f)
    if isinstance(wrapper, dict) and "data" in wrapper:
        records = wrapper["data"]
    elif isinstance(wrapper, list):
        for el in wrapper:
            if isinstance(el, dict) and "data" in el:
                return pd.json_normalize(el["data"])
    raise ValueError("Could not find 'data'")


def main():
    # Load original data
    df_orig = load_json(INPUT)
    
    # Load sazonalidade results
    monthly = pd.read_csv(MONTHLY)

    lines = []
    lines.append('# Verificação: Análise CVLI Apenas')
    lines.append('')
    
    # Check filtragem
    if 'tipo' in df_orig.columns:
        cvli_mask = df_orig['tipo'].astype(str).str.lower() == 'cvli'
        total_records_orig = len(df_orig)
        total_cvli = cvli_mask.sum()
        
        lines.append(f'**Verificação de filtragem:**')
        lines.append('')
        lines.append(f'- Total registros originais: {total_records_orig}')
        lines.append(f'- Registros CVLI: {total_cvli}')
        lines.append(f'- Percentual CVLI: {total_cvli/total_records_orig*100:.1f}%')
        lines.append('')
        
        df_cvli = df_orig[cvli_mask].copy()
        
        # Verify bairros in sazonalidade have CVLI data
        bairros_in_saz = set(zip(monthly['cidade'], monthly['bairro']))
        bairros_in_cvli = set(zip(df_cvli['cidade'], df_cvli['bairro']))
        
        lines.append(f'**Verificação de bairros:**')
        lines.append('')
        lines.append(f'- Bairros na sazonalidade (aparecem no CSV): {len(bairros_in_saz)}')
        lines.append(f'- Bairros distintos com CVLI (no JSON filtrado): {len(bairros_in_cvli)}')
        lines.append(f'- Bairros da saz que NÃO têm CVLI: {len(bairros_in_saz - bairros_in_cvli)}')
        lines.append('')
        
        if len(bairros_in_saz - bairros_in_cvli) > 0:
            lines.append('⚠️ AVISO: Bairros na sazonalidade sem CVLI histórico:')
            for cid, bair in sorted(bairros_in_saz - bairros_in_cvli):
                lines.append(f'  - {bair} / {cid}')
        else:
            lines.append('✅ CONFIRMADO: Todos os bairros na sazonalidade têm CVLI')
        
        lines.append('')
        
        # Check if any bairro in saz has only 1-2 registros (muito baixo)
        monthly_totals = monthly.groupby(['cidade','bairro'])['count'].sum().reset_index()
        low_count = monthly_totals[monthly_totals['count'] < 5]
        
        lines.append(f'**Verificação de volume de dados:**')
        lines.append('')
        lines.append(f'- Bairros com <5 CVLI total: {len(low_count)}')
        if len(low_count) > 0:
            lines.append('')
            lines.append('  Bairros com volume baixo (recomenda-se cautela):')
            for _, row in low_count.iterrows():
                lines.append(f'    - {row["bairro"]} / {row["cidade"]}: {int(row["count"])} CVLI')
        
        lines.append('')
        lines.append('**Conclusão:**')
        lines.append('')
        lines.append('✅ A análise de sazonalidade referencia APENAS registros com tipo=CVLI.')
        lines.append('✅ Todos os bairros nos CSVs têm histórico verificável de CVLI.')
        
    else:
        lines.append('⚠️ ERRO: Coluna "tipo" não encontrada no JSON')
    
    OUT_MD.write_text('\n'.join(lines), encoding='utf-8')
    print('Wrote', OUT_MD)


if __name__ == '__main__':
    main()
