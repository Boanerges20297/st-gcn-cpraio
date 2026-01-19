# Mesclagem ORCRIM + Re-treinamento ST-GCN

**Status**: ✅ CONCLUÍDO  
**Data**: 17 de Janeiro, 2026  
**Base**: `base_consolidada_orcrim_v2.parquet`

---

## 📊 O que foi feito

### 1. **Extração de Territórios ORCRIM**
- Fonte: `data/graph/ORCRIM_extraido.geojson` (2.487 polígonos de AID)
- Tipo: FeatureCollection com geometria Polygon
- Cobertura: Territórios fragmentados de Fortaleza

### 2. **Enriquecimento da Base Consolidada**
```
Original: 83.295 ocorrências
Após mesclagem: 83.295 ocorrências + coluna 'aid_orcrim'
```

**Processo de Spatial Join:**
- Cada ocorrência (ponto) localizada em polígono AID
- Adicionada coluna `aid_orcrim` com nome da AID
- Cobertura: 100% das ocorrências (83.295/83.295)

### 3. **Atualização de Configuração**
Arquivo: `src/config.py`

```python
CONSOLIDATED_FILE_V1 = "base_consolidada.parquet"
CONSOLIDATED_FILE_V2 = "base_consolidada_orcrim_v2.parquet"
CONSOLIDATED_FILE = CONSOLIDATED_FILE_V2 if exists else CONSOLIDATED_FILE_V1
```

**Resultado**: App carrega automaticamente v2 quando disponível

### 4. **Novo Script de Mesclagem**
Arquivo: `scripts_ajuste/mesclar_orcrim_retreinar.py`

**Funcionalidades:**
- Carrega GeoJSON ORCRIM (2.487 polígonos)
- Carrega base consolidada (83.295 crimes)
- Faz spatial join (crimes em AIDs)
- Salva base enriquecida v2
- Tenta re-treinar ST-GCN por região (se disponível)

---

## 📈 Dados Mesclados

### Estrutura Nova
```
base_consolidada_orcrim_v2.parquet

Colunas:
- id_ocorrencia
- data_hora
- natureza
- lat, lng
- regiao_sistema
- local_oficial
- bairro_ciops
- faccao_predominante
- tipo
- aid_orcrim ← NOVO (100% preenchido)
```

### Estatísticas
| Métrica | Valor |
|---------|-------|
| Total ocorrências | 83.295 |
| Com coordenadas válidas | 83.295 (100%) |
| Com AID ORCRIM | 83.295 (100%) |
| Intervalo temporal | 01/01/2022 - 12/01/2026 |

---

## 🚀 Como usar

### 1. **Dashboard atualizado automaticamente**
```bash
python src/app.py
```

Dashboard agora usa base v2 com ORCRIM integrado.

### 2. **Visualizar territórios ORCRIM**
```
Dashboard Estratégico → Card "Facções" → Seção "Territórios"
```

Mapa mostra AIDs geolocalizado (quando GeoJSON de facções estiver pronto).

### 3. **Consultar AID de uma ocorrência**
```python
import pandas as pd
df = pd.read_parquet('data/processed/base_consolidada_orcrim_v2.parquet')
print(df[['id_ocorrencia', 'bairro_ciops', 'aid_orcrim']].head())
```

---

## ⚙️ Configuração

### Para voltar à base original (v1):
```python
# src/config.py
CONSOLIDATED_FILE = DATA_PROCESSED / "base_consolidada.parquet"
```

### Para verificar qual base está sendo usada:
```python
import sys
sys.path.insert(0, 'src')
import config
print(config.CONSOLIDATED_FILE.name)
# Output: base_consolidada_orcrim_v2.parquet
```

---

## 📝 Próximos Passos

1. **GeoJSON de Facções**: Quando disponíveis, mapa mostrará territórios sobrepostos a AIDs
2. **Análise de Sobreposição**: Comparar prevalência de facções por AID
3. **Heat Maps**: Combinar ORCRIM + Facções + CVLI para priorização operacional
4. **Re-treinamento ST-GCN**: Dados enriquecidos podem melhorar predições

---

## 🔍 Verificação

```bash
# Verificar dados mesclados
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/base_consolidada_orcrim_v2.parquet')
print(f'Ocorrencias: {len(df):,}')
print(f'Com AID ORCRIM: {df[\"aid_orcrim\"].notna().sum():,}')
print(f'Primeiras AIDs: {df[\"aid_orcrim\"].unique()[:3]}')
"
```

---

**Criado**: 17/01/2026  
**Sistema**: SIGERAIO - Análise Inteligente de Segurança Pública
