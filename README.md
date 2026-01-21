# ST-GCN for Ceará Crime Prediction

Spatio-Temporal Graph Convolutional Networks applied to seizure and crime pattern detection in Fortaleza neighborhoods.

## 🎯 Status: Phase 2 Complete - Feature Engineering Done

- ✅ **Phase 1:** Data normalization & deduplication (neighborhood standardization, city validation)
- ✅ **Phase 2:** Feature engineering (temporal features, moving averages, intensity scores)
- 🔄 **Phase 3:** Spatial graph construction & ST-GCN integration (in progress)

---

## 📁 Project Structure

```
st-gcn_cpraio/
├── data/processed/
│   ├── prisoes_normalized_deduplicated.parquet      (51,750 records - MAIN)
│   ├── prisoes_with_features.parquet               (same + 27 new features)
│   ├── feature_metadata.json
│   └── normalization_params_deduplicated.json
├── src/
│   ├── data/
│   │   ├── neighborhood_deduplicator.py
│   │   ├── city_deduplicator.py
│   │   └── ceara_municipalities.py
│   ├── features/
│   │   ├── temporal_features.py
│   │   └── node_matrix.py
│   └── graph/ (Phase 3)
├── scripts/
│   ├── 01_deduplicate_neighborhoods.py
│   ├── 02_normalize_with_deduplication.py
│   ├── 03_deduplicate_cities.py
│   ├── 04_temporal_features.py
│   └── inspect_cities.py
├── docs/
│   ├── CONSOLIDACAO_NORMALIZACAO_FINAL.md
│   ├── QUICK_REFERENCE_DEDUPLICATED_DATA.md
│   ├── FUZZY_MATCHING_DEDUPLICATION_COMPLETE.md
│   └── VERIFICACAO_CidadeOcor_REPORT.md
└── README.md (this file)
```

---

## 🚀 Quick Start

```bash
# Setup
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# Run data pipeline
python scripts/02_normalize_with_deduplication.py
python scripts/04_temporal_features.py

# Load data in Python
import pandas as pd
df = pd.read_parquet('data/processed/prisoes_with_features.parquet')
```

---

## 📊 Data Overview

**Input:** 9,060 seizure operations (2025-2026)  
**Output:** 51,750 records (375 days × 138 neighborhoods)

**Features:** 32 columns
- 3 normalized seizure types (drugs, weapons, money)
- 9 lag features (t-1, t-7, t-30 days)
- 6 moving averages (7-day, 30-day windows)
- 3 volatility measures
- 1 intensity score
- 4 cyclical temporal features (day/month)

---

## 🔧 Core Modules

**`src/data/neighborhood_deduplicator.py`**  
Fuzzy matching for neighborhood name standardization (93% success rate)

**`src/features/temporal_features.py`**  
Lag features, moving averages, intensity scores, volatility, cyclical encoding

**`src/features/node_matrix.py`**  
Convert time-series to tensor format (T=375, N=138 neighborhoods, F=variable)

---

## 📈 Data Quality

✅ No NaN values  
✅ All normalized features in [0.0, 1.0]  
✅ 100% temporal coverage (375 consecutive days)  
✅ Zero duplicate (neighborhood, date) pairs  
✅ 100% neighborhood mapping to official names

---

## 📝 Documentation

| File | Content |
|------|---------|
| [CONSOLIDACAO_NORMALIZACAO_FINAL.md](docs/CONSOLIDACAO_NORMALIZACAO_FINAL.md) | Phase 1 summary & metrics |
| [QUICK_REFERENCE_DEDUPLICATED_DATA.md](docs/QUICK_REFERENCE_DEDUPLICATED_DATA.md) | How to use the dataset |
| [FUZZY_MATCHING_DEDUPLICATION_COMPLETE.md](docs/FUZZY_MATCHING_DEDUPLICATION_COMPLETE.md) | Technical details |
| [VERIFICACAO_CidadeOcor_REPORT.md](docs/VERIFICACAO_CidadeOcor_REPORT.md) | City validation report |

---

## 🔜 Phase 3

- [ ] Build spatial adjacency matrix
- [ ] Construct graph edge indices  
- [ ] Build node feature tensors
- [ ] Validate tensor shapes
- [ ] Integrate PyTorch Geometric
- [ ] Train ST-GCN
   ```

2. Testar filtro de data no dashboard

3. Validar priorização de CVLI no mapa

## ⚙️ Configuração CVLI

A configuração de prioridade de crimes violentos letais está em `src/config.py`:

```python
class HyperParams:
    cvli_weight: float = 5.0  # Multiplicador para crimes letais
```

Este peso é aplicado em:
- Cálculos de risco
- Visualização de mapas (3x mais intenso)
- Análise estratégica da IA

---

**Última atualização**: Janeiro 17, 2026  
**Versão**: 1.1.0  
Veja [CHANGELOG.md](docs/CHANGELOG.md) para histórico completo.
