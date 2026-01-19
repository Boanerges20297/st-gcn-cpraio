<!-- ORGANIZE_SCRIPTS_REMINDER -->
# 📋 ORGANIZAÇÃO DE SCRIPTS - REFERÊNCIA

## ✅ SCRIPTS EM `/scripts_ajuste` (Correto)

Todos os scripts de **ajuste, manutenção, testes e análise** devem estar em `/scripts_ajuste/`:

### 🔧 Scripts de Ajuste/Manutenção
- ✓ `analisar_orcrim.py` - Análise de dados ORCRIM
- ✓ `criar_mapa_territorial.py` - Geração de mapas por facção
- ✓ `criar_predicoes_bairros.py` - Geração de predições por bairro
- ✓ `entender_orcrim.py` - Exploração de estrutura de dados
- ✓ `preview_dashboard.py` - Preview dos dashboards
- ✓ `resumo_integracao_visual.py` - Resumo visual de integração
- ✓ `strategic_analyzer.py` - Análise estratégica de dados

### 🧪 Scripts de Teste
- ✓ `test_backend.py` - Testes do backend Flask
- ✓ `test_bairro_predictions.py` - Testes de predições por bairro
- ✓ `test_dashboard_bairros.py` - Testes do dashboard de bairros
- ✓ `test_dashboard_routes.py` - Testes de rotas do dashboard
- ✓ `test_integracao_completa.py` - Testes de integração completa
- ✓ `test_integration_bairros.py` - Testes de integração de bairros
- ✓ `test_predicao.py` - Testes de predição
- ✓ `test_territorios.py` - Testes de dados territoriais

---

## 📁 SCRIPTS NA RAIZ (Aplicação Principal)

Esses scripts estão na raiz PORQUE são pontos de entrada da aplicação:

| Script | Propósito | Localização |
|--------|-----------|------------|
| `main.py` | Treina o modelo e gera predições | ✓ Raiz `/` |
| `run_app.py` | Inicia servidor Flask | ✓ Raiz `/` |

---

## 📌 REGRA GERAL

```
Novo script criado? Pergunte-se:
├─ É ajuste/teste/manutenção? → Va para /scripts_ajuste/
├─ É ponto de entrada da app? → Deixe na raiz
└─ É módulo utilitário? → Va para /src/
```

---

## 🚀 Como Executar Scripts de Ajuste

```bash
# De qualquer diretório:
cd projeto-stgcn-cpraio
python scripts_ajuste/nome_do_script.py

# Ou manualmente:
cd scripts_ajuste/
python nome_do_script.py
```

---

**Última atualização:** 17/01/2026
