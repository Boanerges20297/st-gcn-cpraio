# 📋 REVISÃO: ARQUITETURA DE PREDIÇÃO DO MODELO

**Data:** 20/01/2026  
**Assunto:** Análise dos 3 pontos críticos da estratégia de predição

---

## 1️⃣ JANELA DE PREDIÇÃO COM CRÍTICO 180d + CVLI

### Estado Atual:
```
┌────────────────────────────────────┐
│ HISTÓRICO (Input)                  │
├────────────────────────────────────┤
│ Período: 14 dias de histórico      │
│ Granularidade: Diário              │
│ Nós: 319 (bairros/municípios)      │
│ Features: 6 (CVLI, CVP, CV, etc)   │
│ Normalização: (X - μ) / σ          │
└────────────────────────────────────┘
              ↓ [ST-GCN]
┌────────────────────────────────────┐
│ PREDIÇÃO (Output)                  │
├────────────────────────────────────┤
│ Horizonte: 15 dias (quinzena)      │
│ Saída: Média agregada              │
│ Formato: (1 nó, 6 features)        │
│ Desnormalização: X' * σ + μ        │
└────────────────────────────────────┘
```

### Características da Janela 180d:

**Arquivo responsável:** [src/config.py](src/config.py#L72)
```python
HyperParams = {
    'window_size': 14,    # 14 dias de histórico (INPUT)
    'target_window': 15,  # 15 dias de predição  (OUTPUT)
    'hidden_dim': 32,
    'batch_size': 32,
    'epochs': 200,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'dropout': 0.4,
    'cvli_weight': 5.0    # ⭐ PONDERAÇÃO CVLI NO TREINO
}
```

**Lógica de predição:** [main.py](main.py#L52-L75)
- Coleta **últimos 14 dias** de histórico → normaliza com μ/σ do treino
- Passa pelo modelo ST-GCN → gera representação latente
- Output: Previsão média para os **próximos 15 dias**
- Desnormaliza e aplica ReLU (sem negativos)

**ETL responsável:** [scripts_ajuste/01_etl_janela180d_otimizado.py](scripts_ajuste/01_etl_janela180d_otimizado.py)
- Janela móvel de 180 dias
- Filtra apenas CVLI (crítico)
- Treino: 80% dos dias | Validação: 20%

### ✅ O que funciona:
- ✓ Filtragem CVLI-only no ETL
- ✓ Ponderação CVLI (`cvli_weight=5.0`) durante treino
- ✓ Normalização consistente (treino ≠ predição)
- ✓ Datas calculadas corretamente (último_dia + 1 até + 15)

### ⚠️ Limitações atuais:
- ❌ **Saída é MÉDIA agregada** → perde variabilidade diária
- ❌ Janela de 180d é **fixa** → não é configurável
- ❌ **Sem intervalos de confiança** → apenas ponto central
- ❌ Predição sempre **15 dias** → não é parametrizável

---

## 2️⃣ MODELO TREINADO 2022-2025 → PREDIZ +15 DIAS DA ÚLTIMA OCORRÊNCIA

### Timeline Atual:

```
┌──────────────────────────────────────────────────────────┐
│ DADOS DE TREINO                                          │
├──────────────────────────────────────────────────────────┤
│ 2022-01-01 ┌─────────────────────────────────────┐      │
│            │    HISTÓRICO COMPLETO: 1461 DIAS   │      │
│ 2024-12-31 │ (3 anos de ocorrências policiais)  │      │
│            └─────────────────────────────────────┘      │
│                         ↓ [TREINO]                       │
│           Split: 80% treino | 20% validação            │
│                                                          │
│ Modelo salvo: model_janela180d.pth                      │
│ Stats: {mean, std} da distribuição 2022-2024           │
└──────────────────────────────────────────────────────────┘
                         ↓ [INFERÊNCIA]
┌──────────────────────────────────────────────────────────┐
│ PREDIÇÃO 2025+                                           │
├──────────────────────────────────────────────────────────┤
│ Entrada: Últimos 14 dias de 2025                        │
│ Data base: Ex: 20/01/2025 (última ocorrência)           │
│ Saída: Predição para 21/01 ~ 04/02/2025               │
│                                                          │
│ Relógio Tático:                                         │
│   └─ Data base: DD/MM/YYYY                              │
│   └─ Início predição: +1 dia                            │
│   └─ Fim predição: +15 dias                             │
└──────────────────────────────────────────────────────────┘
```

### ✅ Como funciona hoje:

**Arquivo:** [main.py:predict_future()](main.py#L52-L75)
```python
# 1. Pega últimos 14 dias de X_full
last_window = X_full[-window_size:]  # window_size=14

# 2. Normaliza com estatísticas do treino
last_window_norm = (last_window - mean) / std

# 3. Passa pelo modelo (inferência)
prediction_norm = model(input_tensor, edge_index)

# 4. Desnormaliza
prediction_real = prediction_norm * std + mean

# 5. Calcula horizonte: último_dia + 15 dias
pred_start = last_date_obj + timedelta(days=1)
pred_end = last_date_obj + timedelta(days=target_window)
```

### ⚠️ Questão: "O modelo é capaz de prever 'amanhã'?"

**RESPOSTA:** Teoricamente SIM, mas com ressalvas:

| Aspecto | Status | Detalhes |
|---------|--------|----------|
| **Arquitetura permite?** | ✅ SIM | ST-GCN pode fazer step-by-step |
| **Modelo treinado assim?** | ❌ NÃO | Treina para média de 15 dias (agregada) |
| **Saída atual** | 📊 Média | Valor único para 15 dias = `E[crime_t+1...t+15]` |
| **Pode modificar?** | ✅ POSSÍVEL | Requer mudança no target do treino |

---

## 3️⃣ MODELO PODE SER PERSONALIZADO? (AMANHÃ? SEMANA?)

### Limitação Atual: **Saída é Fixa em 15 Dias**

Hoje o fluxo é:

```
Script: main.py ou src/predict.py
         └─ Sempre chama predict_future()
            └─ HARDCODED: target_window=15
               └─ Saída: predição média para próximos 15 dias
```

### 🎯 Solução Proposta: PARAMETRIZÁVEL

**Opção 1: Predição Daily (Dia a Dia)**
```
Input: Últimos 14 dias
Modelo: Treinado com target_window=1 (requer RETREINO)
Output: Predição para AMANHÃ (24h)
```

**Opção 2: Predição Flexível (1-30 dias)**
```
Input: Últimos 14 dias
Modelo: Armazena sequência completa de predições
Output: Customizável via parâmetro (7d, 15d, 30d)
```

**Opção 3: Predição Probabilística (Intervalo de Confiança)**
```
Input: Últimos 14 dias
Modelo: Usa ensemble de modelos
Output: [valor_baixo, valor_médio, valor_alto] + confiança
```

### ❌ Bloqueios para Implementação Hoje:

| Bloqueio | Impacto | Solução |
|----------|---------|---------|
| Modelo retornando **média** não séries | 🔴 CRÍTICO | Retrainer com `target_window=1` |
| Parâmetros hardcoded em `config.py` | 🟡 MÉDIO | Mover para arquivo JSON/YAML |
| Sem suporte a **inference dinamicamente** | 🟡 MÉDIO | Adicionar função `predict_custom_days()` |
| Treino feito apenas para **15 dias** | 🔴 CRÍTICO | Preparar múltiplos modelos |

---

## 📊 COMPARATIVO: CENÁRIOS DE USO

### Cenário A: Status Quo (Hoje)
```yaml
Entrada: "20/01/2025"
Janela: 14 dias histórico
Saída: "21/01 a 04/02 = Média 85 CVLIs"
Tipo: Agregado Quinzenal
Confiança: Ponto único
```

### Cenário B: Requerido (Parametrizável)
```yaml
Entrada: "20/01/2025 + horizon=7 (dias)"
Janela: 14 dias histórico
Saída: [21/01→40 CVLI, 22/01→42 CVLI, ..., 27/01→51 CVLI]
Tipo: Diário desagregado
Confiança: Intervalo [low, mid, high]
```

### Cenário C: Avançado (Futuro)
```yaml
Entrada: "20/01/2025 + model='week' + confidence=0.95"
Janela: 14 dias histórico
Saída: "21/01-27/01 = [200-250 CVLI] (95% confiança)"
Tipo: Semanal com incerteza
Confiança: Intervalos de confiança
```

---

## 🔧 RECOMENDAÇÕES TÉCNICAS

### 1. Curto Prazo (Sem Retreino)
**Objetivo:** Tornar parâmetros flexíveis SEM alterar modelo

```python
# Novo arquivo: src/predict_parametrizado.py

def predict_custom(
    model, 
    X_full, 
    edge_index, 
    mean, 
    std,
    prediction_days: int = 15,  # ⭐ CUSTOMIZÁVEL
    start_date: str = None,     # ⭐ CUSTOMIZÁVEL
    interval_type: str = 'aggregated'  # 'daily' | 'weekly' | 'aggregated'
):
    """
    Predição parametrizável sem retreino.
    
    - prediction_days: 1-30 (dias à frente)
    - start_date: 'YYYY-MM-DD' ou None (usar último)
    - interval_type: 'daily' (interpola), 'weekly' (agrupa)
    """
    # Lógica a implementar
    pass
```

### 2. Médio Prazo (Com Retreino Modular)
**Objetivo:** Treinar múltiplos modelos para diferentes horizontes

```
data/models/
├── model_1day.pth       # Predição +1 dia
├── model_7days.pth      # Predição +7 dias
├── model_15days.pth     # Predição +15 dias (ATUAL)
└── model_30days.pth     # Predição +30 dias
```

### 3. Longo Prazo (Arquitetura Seq2Seq)
**Objetivo:** Migrar para modelo capaz de predizer múltiplos passos

```
Arquitetura: Encoder-Decoder com Attention
├── Encoder: 14 dias histórico
├── Decoder: Prediz sequencialmente t+1, t+2, ..., t+N
└── Attention: Aprende dependências temporais
```

---

## 📝 MATRIZ DE DECISÃO

| Funcionalidade | Hoje | Curto Prazo | Médio Prazo | Complexidade |
|---|---|---|---|---|
| Predição 15 dias | ✅ | ✅ | ✅ | Baixa |
| Predição 7 dias | ❌ | ✅ | ✅ | Média |
| Predição 1 dia (daily) | ❌ | 🟡 | ✅ | Alta |
| Intervalos confiança | ❌ | ❌ | ✅ | Muito Alta |
| Parâmetros flexíveis | ❌ | ✅ | ✅ | Média |
| Treino múltiplos horizontes | ❌ | ❌ | ✅ | Muito Alta |

---

## 🎬 PRÓXIMAS AÇÕES SUGERIDAS

### ✅ AÇÃO 1: Implementar Predição Parametrizada (CURTO PRAZO)
```bash
# Arquivo: src/predict_parametrizado.py
# Tempo: 2-3 horas
# Impacto: Permite usar mesma IA com diferentes horizontes
```

### ✅ AÇÃO 2: Documentar Limitações Atuais (IMEDIATO)
```
✓ Dashboard mostra: "Predição: +15 dias a partir de [data]"
✓ API expõe parâmetro: ?prediction_days=15 (ignorado por enquanto)
✓ Roadmap claro para amanhã/semana
```

### ✅ AÇÃO 3: Validação de Performance (MÉDIO PRAZO)
```
✓ Teste: Treinar modelo com target_window=1
✓ Comparar: MSE(1d) vs MSE(15d)
✓ Decidir: Vale a pena retreinar para daily?
```

### ✅ AÇÃO 4: Preparar Arquitetura Seq2Seq (FUTURO)
```
✓ Research: LSTM Encoder-Decoder + Attention
✓ POC: Implementar versão experimental
✓ Benchmark: Comparar com STGCN atual
```

---

## 📌 CONCLUSÃO

| Pergunta | Resposta | Contexto |
|----------|----------|----------|
| **Tem janela 180d + CVLI?** | ✅ SIM | Implementado em scripts_ajuste |
| **Model treinado 2022-2025?** | ✅ SIM | 1461 dias de histórico |
| **Prevê +15 dias?** | ✅ SIM | `target_window=15` em config.py |
| **Pode prever amanhã?** | 🟡 TEORICAMENTE | Requer mudança no target do treino |
| **Pode ser personalizado?** | 🟡 POSSÍVEL | Requer refatoração modular |
| **Suporta semana?** | ❌ HOJE | Pode ser implementado sem retreino |

**Recomendação:** Implementar **Ação 1** (predição parametrizada) + **Ação 2** (documentar), depois validar se **Ação 3** vale a pena.

