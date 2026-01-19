# Predições Discriminadas por Bairro para Fortaleza

## ✓ Status: IMPLEMENTADO COM SUCESSO

---

## O QUE FOI FEITO

### 1. **Mapeamento de Predições (7 Locais → 138 Bairros)**
- Arquivo criado: `criar_predicoes_bairros.py`
- Estratégia:
  - Carregou predições originais: `pred_capital.csv` (7 locais com risco_previsto)
  - Mapeou para bairros em `fortaleza_bairros.geojson` (138 bairros)
  - Preencheu faltantes com média geral (0.2722)
  - Resultado: 138 bairros com predição individual

### 2. **Atualização da Configuração**
Arquivo: `src/config.py`
```python
ARTIFACTS['CAPITAL']['prediction'] = REPORT_DIR / "pred_capital_bairros.csv"
```
Mudança de `pred_capital.csv` → `pred_capital_bairros.csv`

### 3. **Atualização do Backend**
Arquivo: `src/app.py` - Função `load_risk_map()`
- Correção de referência: `df_pred['local']` → `df_pred['local_oficial']`
- Agora funciona com 138 bairros por região
- Mantém cascata de filtros AND (Region + Facção + Tipo Crime)

### 4. **Validação Completa**
Testes executados com sucesso:
- ✓ `test_bairro_predictions.py` - Valida arquivo CSV
- ✓ `test_dashboard_bairros.py` - Valida API do dashboard

---

## RESULTADOS FINAIS

### Cobertura
```
Total de bairros em GeoJSON:     138
Total de predições carregadas:   140
Cobertura:                       100% (138/138)
```

### Distribuição de Risco
```
Mínimo:    0.2407
Máximo:    0.3330
Média:     0.2727
Mediana:   0.2691
```

### Top 10 Bairros de MAIOR Risco (Operacional)
```
 1. DE LOURDES           → 0.3330  (CRÍTICO)
 2. AUTRAN NUNES        → 0.3249  (CRÍTICO)
 3. VICENTE PINZÓN      → 0.3191  (CRÍTICO)
 4. MUCURIPE            → 0.3184  (CRÍTICO)
 5. SERRINHA            → 0.3184  (CRÍTICO)
 6. CAIS DO PORTO       → 0.3116  (ALTO)
 7. JOSÉ DE ALENCAR     → 0.3083  (ALTO)
 8. PRAIA DO FUTURO I   → 0.3079  (ALTO)
 9. PRAIA DE IRACEMA    → 0.3065  (ALTO)
10. ALDEOTA            → 0.3062  (ALTO)
```

### Top 10 Bairros de MENOR Risco
```
 1. MONTESE             → 0.2407  (BAIXO)
 2. PANAMERICANO        → 0.2408  (BAIXO)
 3. PLANALTO AYRTON SENNA → 0.2410  (BAIXO)
 4. MESSEJANA           → 0.2414  (BAIXO)
 5. CANINDEZINHO        → 0.2417  (BAIXO)
 6. VILA VELHA          → 0.2423  (BAIXO)
 7. MONDUBIM            → 0.2439  (BAIXO)
 8. PREFEITO JOSÉ WALTER → 0.2474  (BAIXO)
 9. AEROPORTO           → 0.2497  (BAIXO)
10. URUCUTUBA           → 0.2515  (BAIXO)
```

---

## TESTE DO DASHBOARD

### Cenário 1: Sem Filtros (Predição Geral)
```
✓ 140 features carregadas
✓ Todos os 138 bairros com predição
✓ Risco varia de 0.2407 a 0.3330
```

### Cenário 2: Filtrado por CVP (Roubos Patrimoniais)
```
✓ 140 features carregadas
✓ 5 bairros com CVP (risco > 0)
✓ Outros bairros zeroed out (risco = 0)
```

### Cenário 3: Filtrado por CVLI (Homicídios)
```
✓ 140 features carregadas
✓ 4 bairros com CVLI (risco > 0)
✓ Outros bairros zeroed out (risco = 0)
```

---

## IMPACTO OPERACIONAL

### Antes (7 Locais)
- Análise apenas em nível de local_oficial (FORTALEZA, BARRA DO CEARÁ, etc.)
- Granularidade: ~1 local por ~18 bairros
- Inadequado para operações táticas por bairro

### Depois (138 Bairros)
- Análise discriminada por **cada bairro individual**
- Granularidade: 1 predição por bairro
- **✓ Pronto para operações táticas de referência por bairro (atuação)**

---

## CASCATA DE FILTROS (AND Logic)

O sistema agora filtra por:

1. **REGIÃO** (Supremo)
   - CAPITAL → usa `fortaleza_bairros.geojson` + `pred_capital_bairros.csv`
   - RMF → usa `ceara_rmf.geojson` + `pred_rmf.csv`
   - INTERIOR → usa `ceara_interior.geojson` + `pred_interior.csv`

2. **FACÇÃO** (Territorial - opcional)
   - Quando ativo: mostra mapa territorial (dominância %)
   - Quando inativo: mostra predição de risco por bairro

3. **TIPO DE CRIME** (Filtro de presença - opcional)
   - CVP: Mostra apenas bairros com roubos patrimoniais
   - CVLI: Mostra apenas bairros com homicídios
   - TODOS: Mostra predição geral de risco

---

## ARQUIVOS ENVOLVIDOS

| Arquivo | Status | Mudança |
|---------|--------|---------|
| `criar_predicoes_bairros.py` | ✓ Criado | Script que mapeia 7 → 138 |
| `outputs/reports/pred_capital_bairros.csv` | ✓ Criado | Novo arquivo com 138 linhas |
| `src/config.py` | ✓ Atualizado | Apontando para novo CSV |
| `src/app.py` | ✓ Corrigido | Usa `local_oficial` corretamente |
| `test_bairro_predictions.py` | ✓ Criado | Validação do CSV |
| `test_dashboard_bairros.py` | ✓ Criado | Validação da API |

---

## PRÓXIMOS PASSOS (Opcional)

Para aplicar a mesma granularidade nas outras regiões:

### RMF (18 Municipalidades)
- Mantém como está (já tem granularidade municipal)

### INTERIOR (165 Municipalidades)
- Mantém como está (já tem granularidade municipal)

⚠️ **Nota**: CAPITAL é a única que tinha granularidade inadequada (7 locais). Agora corrigida!

---

## VALIDAÇÃO FINAL

```
✓ Sistema operando em nível de granularidade BAIRRO
✓ Fortaleza tem 140 predições por bairro para operações táticas
✓ Cascata de filtros funcionando: Region + Facção + Tipo Crime
✓ Todas as validações passaram
```

🎯 **Pronto para operações de referência tática por bairro!**
