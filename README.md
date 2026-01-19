# PROJETO STGCN - CPRAIO

Sistema de análise e predição de crimes com ST-GCN (Spatial-Temporal Graph Convolutional Networks) para gestão estratégica de segurança pública.

## 📚 Documentação

Toda documentação está organizada em [`docs/`](docs/):

- **[INDICE_DOCUMENTACAO.md](docs/INDICE_DOCUMENTACAO.md)** - Navegação completa por tipo de leitor (gestor, dev, QA, DevOps)
- **[MAPA_MENTAL_AJUSTES.md](docs/MAPA_MENTAL_AJUSTES.md)** - Visão geral dos ajustes implementados (janeiro 2026)
- **[IMPLEMENTACOES_17JAN2026.md](docs/IMPLEMENTACOES_17JAN2026.md)** - Detalhes técnicos de cada implementação
- **[RESUMO_VISUAL_AJUSTES.md](docs/RESUMO_VISUAL_AJUSTES.md)** - Diagramas, flowcharts e comparações visuais
- **[CHANGELOG.md](docs/CHANGELOG.md)** - Histórico de versões e mudanças
- **[SUMARIO_EXECUTIVO_AJUSTES.md](docs/SUMARIO_EXECUTIVO_AJUSTES.md)** - Resumo executivo para tomadores de decisão
- **[ORGANIZACAO_SCRIPTS.md](docs/ORGANIZACAO_SCRIPTS.md)** - Documentação de scripts de ajuste

## 🚀 Quick Start

```bash
# Instalar dependências
pip install -r requirements.txt

# Iniciar aplicação
python src/app.py

# Ou via script seguro (sem auto-reload)
python run_app.py
```

Dashboard disponível em: `http://localhost:5000/dashboard-estrategico`

## 📁 Estrutura do Projeto

```
projeto-stgcn-cpraio/
├── src/                    # Código-fonte principal
│   ├── app.py             # Flask application
│   ├── config.py          # Configuração (CVLI weight: 5.0)
│   ├── model.py           # ST-GCN neural network
│   ├── predict.py         # Predições de crime
│   └── templates/         # HTML dashboards
├── scripts_ajuste/        # Scripts de ajuste/manutenção
│   └── integrar_faccoes_geojson.py  # Integração de dados de facção
├── data/                  # Dados (cache, processed, raw, tensors, graph)
├── notebooks/             # Análises exploratórias Jupyter
├── docs/                  # 📚 DOCUMENTAÇÃO (LEIA AQUI!)
├── outputs/               # Relatórios, mapas, modelos
└── requirements.txt       # Dependências Python
```

## ✅ Configuração Atual

- **CVLI Priority**: ✅ Implementado (weight: 5.0x)
- **Date Filter**: ✅ Implementado (UI + Backend route `/api/strategic_insights_range`)
- **Faction Geolocation**: ✅ Script criado (`integrar_faccoes_geojson.py`)
- **Scripts Organization**: ✅ Todos em `/scripts_ajuste/`
- **Documentation**: ✅ Centralizada em `/docs/`

## 🔧 Guias Práticos

Para começar rapidamente:
1. Leia [INDICE_DOCUMENTACAO.md](docs/INDICE_DOCUMENTACAO.md) conforme seu perfil
2. Para visão geral: [MAPA_MENTAL_AJUSTES.md](docs/MAPA_MENTAL_AJUSTES.md)
3. Para integração: Acesse `/scripts_ajuste/integrar_faccoes_geojson.py`

## 📋 Próximos Passos

1. Executar script de integração de facção:
   ```bash
   python scripts_ajuste/integrar_faccoes_geojson.py
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
