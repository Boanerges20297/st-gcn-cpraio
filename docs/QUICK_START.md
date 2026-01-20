# ✅ CHECKLIST DE INÍCIO RÁPIDO

**Seu novo modelo ST-GCN com parâmetros avançados está pronto para ser construído.**

Este documento orienta você através dos primeiros passos para iniciar a implementação.

---

## 🎯 FASE 0: VALIDAÇÃO (1-2 dias)

Antes de começar a codificar, valide estas suposições:

### Validação de CVLI
- [ ] **Agendar com:** Delegacia, CPRAIO ou especialista em crimes violentos
- [ ] **Confirmar:** Lista exata de crimes que contam como CVLI
- [ ] **Exemplo esperado:**
  ```
  ✅ CVLI: HOMICÍDIO, ESTUPRO, ROUBO, LESÃO CORPORAL GRAVE
  ❌ NÃO-CVLI: FURTO, POSSE DE DROGA, TRÁFICO (mesmo relevante)
  ```
- [ ] **Documentar:** Salvar definição oficial em `docs/CVLI_DEFINICAO_OFICIAL.md`

### Validação de Territorios Faccionados
- [ ] **Confirmar:** Que os GeoJSON em `data/raw/inteligencia/` estão atualizados
- [ ] **Validar:** Que cobrem toda região de interesse (Capital, RMF, Interior)
- [ ] **Checar:** Se há novas facções não mapeadas

### Validação de Dados de Prisões
- [ ] **Confirmar:** Que `ocorrencias_tropa.json` é a fonte correta
- [ ] **Revisar:** Amostra de 10 registros manualmente
- [ ] **Validar:** Que coordenadas estão em formato DMS ou decimal
- [ ] **Estimar:** Qualidade de parsing esperada (melhor ~95%)

### Aprovação de Stakeholders
- [ ] **Apresentar:** Sumário da viabilidade para CPRAIO
- [ ] **Obter:** Aprovação formal para implementação
- [ ] **Alinhar:** Cronograma e expectativas

---

## 🛠️ FASE 1: SETUP DO AMBIENTE (1 dia)

### Dependências Python
```bash
# Ativar seu venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\Activate.ps1  # Windows PowerShell

# Instalar pacotes adicionais se necessário
pip install geopandas scipy scikit-learn tqdm

# Verificar versões
python -c "import torch; import pandas; print(torch.__version__)"
```

### Estrutura de Pastas
```bash
mkdir -p docs/analises docs/modelos outputs/v2_predictions

# Confirmar estrutura
ls -la data/raw/                    # Deve ter ocorrencia_policial_operacional.json
ls -la data/raw/inteligencia/       # Deve ter GeoJSON de facções
ls -la data/processed/              # Deve ter base_consolidada_orcrim_v3.parquet
```

### Backup de Código Atual
```bash
# Fazer backup do model.py, trainer.py antes de modificar
cp src/model.py src/model_backup_$(date +%Y%m%d).py
cp src/trainer.py src/trainer_backup_$(date +%Y%m%d).py
cp src/graph_builder.py src/graph_builder_backup_$(date +%Y%m%d).py
```

---

## 📝 FASE 2: IMPLEMENTAÇÃO (17-20 dias)

### Sprint 1: Normalização de Dados (5-7 dias)

**Objetivo:** Criar arquivo `data/unified_2025.parquet` com dados limpos

**Checklist:**
- [ ] **Task 1.1:** Implementar `parse_operational_json()` em `data_loader.py`
  - Extrair lat/long corretamente
  - Criar coluna `is_cvli`
  - Testar: `df_op = parse_operational_json(...); print(df_op.head())`

- [ ] **Task 1.2:** Implementar `parse_tropa_narrative()` em `data_loader.py`
  - Regex para extrair informações de texto
  - Testar com amostra de 10 narrativas manualmente
  - Verificar taxa de sucesso de parsing

- [ ] **Task 1.3:** Criar `normalize_tropa_dataset()`
  - Converter coordenadas DMS → decimal
  - Vincular com dados operacionais por data/local
  - Salvar `data/unified_2025.parquet`

**Saída esperada:**
```python
df_unified.shape  # (~9500, 20 columns)
df_unified.columns  # id, data, municipio, ..., is_cvli, has_large_seizure, has_weapons_drugs
df_unified['is_cvli'].sum()  # ~800-1000 crimes CVLI
```

**Validação:**
```python
# Rodar script de validação
python -c "
import pandas as pd
df = pd.read_parquet('data/unified_2025.parquet')
print(f'Total: {len(df)}')
print(f'CVLI: {df[\"is_cvli\"].sum()}')
print(f'Apreensões >= 1kg: {df[\"has_large_seizure\"].sum()}')
print(f'Arma+droga: {df[\"has_weapons_drugs\"].sum()}')
print(df.info())
"
```

---

### Sprint 2: Feature Engineering (3-4 dias)

**Objetivo:** Criar tensor X_extended com 7 features por nó/dia

**Checklist:**
- [ ] **Task 2.1:** Atualizar `config.py`
  - Adicionar `CVLI_DEFINITIONS` dict
  - Adicionar `CRIME_WEIGHTS` dict
  - Definir `X_FEATURE_DIMENSIONS`

- [ ] **Task 2.2:** Modificar `graph_builder.py`
  - Criar função `build_graph_with_weights()`
  - Implementar ponderação de edges (CVLI 3x, drogas 2x, etc)
  - Retornar X_extended com 7 features

- [ ] **Task 2.3:** Testar construção de grafo
  ```python
  from graph_builder import build_graph_with_weights
  graph = build_graph_with_weights('CAPITAL', df_unified)
  print(f"X shape: {graph['X'].shape}")  # Deve ser (num_days, num_nodes, 7)
  print(f"Edge weights: min={graph['edge_weight'].min()}, max={graph['edge_weight'].max()}")
  ```

**Saída esperada:**
```python
X.shape  # (num_days, num_nodes, 7)
edge_weight.unique()  # Deve ter valores > 1.0 onde CVLI/drogas/armas estão
```

---

### Sprint 3: Adaptação do Modelo (3-4 dias)

**Objetivo:** Treinar modelo v2 com novas features

**Checklist:**
- [ ] **Task 3.1:** Criar `STGCN_Cpraio_v2` em `model.py`
  - Aceitar `in_channels` variável (7 em vez de 1)
  - Adicionar `edge_weight` parameter

- [ ] **Task 3.2:** Criar `train_region_v2()` em `trainer.py`
  - Usar `build_graph_with_weights()`
  - Usar 250 épocas (em vez de 200)
  - Salvar stats por feature

- [ ] **Task 3.3:** Testar treinamento
  ```python
  from trainer import train_region_v2
  train_region_v2('CAPITAL', df_unified)
  # Deve levar 30-60 minutos (com GPU)
  ```

**Saída esperada:**
```
Epoch 10/250 | Train: 0.1234 | Val: 0.1456
Epoch 20/250 | Train: 0.0987 | Val: 0.1123
...
[✓] Modelo salvo: outputs/models/model_capital_v2.pth
```

---

### Sprint 4: Validação (2-3 dias)

**Objetivo:** Validar eficácia do novo modelo em dados reais 2025

**Checklist:**
- [ ] **Task 4.1:** Implementar função de backtest
  ```python
  def backtest_2025(model, df_unified, train_end='2025-08-31', test_end='2025-11-30'):
      # Treinar com dados até 2025-08-31
      # Prever de 2025-09-01 até 2025-11-30
      # Comparar com real
      # Retornar métricas
      pass
  ```

- [ ] **Task 4.2:** Calcular métricas
  - RMSE: Erro quadrático médio
  - MAE: Erro absoluto médio
  - R²: Coeficiente de determinação
  - Correlação com prisões (novo!)

- [ ] **Task 4.3:** Gerar relatório
  ```python
  # Salvar resultados em CSV e visualizar
  results.to_csv('outputs/backtest_results_2025.csv')
  
  # Plot: Predito vs Real
  plt.plot(dates, real_crimes, label='Real', alpha=0.7)
  plt.plot(dates, predicted_crimes, label='Predito', alpha=0.7)
  plt.legend()
  plt.savefig('outputs/backtest_comparison.png', dpi=150)
  ```

**Saída esperada:**
```
=== BACKTEST 2025 ===
RMSE: 12.34
MAE: 8.76
R²: 0.85
Correlation(Prisões, CrimeReduction): 0.67 (significa: prisões reduzem crimes!)
```

---

## ✅ FASE 3: VALIDAÇÃO FINAL (1-2 dias)

Após completar todas as tarefas:

- [ ] **Código review:** Revisar com colega
- [ ] **Testes unitários:** Confirmar que não há bugs
- [ ] **Documentação:** Atualizar READMEs e comentários
- [ ] **Performance:** Confirmar que roda em tempo aceitável
- [ ] **Apresentação:** Preparar deck executivo com resultados

---

## 📊 INDICADORES DE SUCESSO

### Ao final das implementações, você deve ter:

| Métrica | Esperado | Seu Resultado |
|---------|----------|---------------|
| Dataset unificado carregável | ✅ Sim | — |
| CVLI identificados corretamente | ~800-1000 | — |
| Tensor com 7 features | (num_days, nodes, 7) | — |
| Modelo treina sem erros | ✅ Sim | — |
| RMSE <= 15 | ✅ Sim | — |
| R² >= 0.80 | ✅ Sim | — |
| Correlação prisões vs crimes | > 0.50 | — |

---

## 🆘 TROUBLESHOOTING

### Erro ao parsear JSON operacional
```python
# Problema: "JSONDecodeError"
# Solução:
import json
with open('file.json', 'r', encoding='utf-8-sig') as f:  # Tentar encoding diferente
    data = json.load(f)
```

### Edge weights não sendo usados
```python
# Problema: GCN ignora edge_weight
# Solução: Confirmar que está passando edge_weight para forward()
y_pred = model(x_batch, edge_index_dev, edge_weight_dev)  # ← edge_weight_dev aqui
```

### Coordenadas DMS não convertendo
```python
# Problema: "-5°15'53.4"S" não converte para decimal
# Solução: Usar libraria dms2dd
from dms2dd import parse
lat = parse("-5°15'53.4\"S")  # Retorna -5.264833
```

### Treinamento muito lento
```python
# Problema: 250 épocas levando horas
# Solução: Usar GPU
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando: {device}")  # Deve aparecer "cuda"
```

---

## 📚 RECURSOS

### Documentação completa:
- `docs/VIABILIDADE_NOVO_MODELO_PARAMETROS.md` — Análise detalhada
- `docs/GUIA_TECNICO_IMPLEMENTACAO.md` — Código e exemplos
- `docs/SUMARIO_VIABILIDADE_2026.md` — Sumário executivo

### Arquivos a modificar:
- `src/config.py` — Adicionar constantes
- `src/data_loader.py` — Adicionar parsers
- `src/graph_builder.py` — Função com pesos
- `src/model.py` — Versão v2
- `src/trainer.py` — Treino atualizado

### Dados de entrada:
- `data/raw/ocorrencia_policial_operacional.json`
- `data/raw/ocorrencias_tropa.json`
- `data/raw/inteligencia/*.geojson`
- `data/processed/base_consolidada_orcrim_v3.parquet`

---

## 🚀 COMO COMEÇAR AGORA

### Opção A: Via Terminal
```bash
# 1. Clonar documentação
cd ~/st-gcn_cpraio
git pull origin

# 2. Ler os docs
cat docs/SUMARIO_VIABILIDADE_2026.md | less

# 3. Começar Sprint 1
python -c "from src import data_loader; help(data_loader.parse_operational_json)"
```

### Opção B: Via VS Code
```bash
# 1. Abrir workspace
code ~/st-gcn_cpraio/

# 2. Abrir docs:
# - docs/VIABILIDADE_NOVO_MODELO_PARAMETROS.md
# - docs/GUIA_TECNICO_IMPLEMENTACAO.md

# 3. Criar arquivo: src/data_loader_v2.py (começar com Task 1.1)
```

---

## ⏰ TIMELINE RECOMENDADA

```
19-01 (Hoje): ✅ Ler este documento
20-01 to 21-01: FASE 0 - Validação (CVLI, Territories, Approvals)
22-01 to 31-01: FASE 1 - Sprint 1 (Normalização dados)
01-02 to 07-02: FASE 2 - Sprint 2 (Features)
08-02 to 14-02: FASE 3 - Sprint 3 (Modelo)
15-02 to 17-02: FASE 4 - Sprint 4 (Validação)
18-02: ✅ MODELO V2 PRONTO EM PRODUÇÃO
```

---

## 📞 SUPORTE

Se tiver dúvidas durante a implementação:

1. **Consulte primeiro:** Os guias técnicos em `docs/`
2. **Procure padrão:** Seu código deve seguir estilos existentes em `src/`
3. **Teste pequeno:** Implemente e teste incrementalmente
4. **Documente:** Adicione comentários explicando lógica complexa

---

## ✨ PRÓXIMO PASSO

**→ Leia: `docs/SUMARIO_VIABILIDADE_2026.md`**  
**→ Depois: `docs/GUIA_TECNICO_IMPLEMENTACAO.md`**  
**→ Então: Comece com Task 1.1**

---

**Você tem tudo que precisa. Bom trabalho! 🚀**

*Documentação gerada: 19-01-2026*  
*Status: READY FOR IMPLEMENTATION*
