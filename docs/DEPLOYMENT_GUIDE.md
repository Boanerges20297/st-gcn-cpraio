# 🚀 GUIA DE DEPLOYMENT - ST-GCN COM DINÂMICA DE FACÇÕES

**Versão:** 2.0 com Dinâmica de Facções  
**Data:** 23 de Janeiro, 2026  
**Status:** Pronto para Produção

---

## 📋 QUICK START

### Fazer uma Predição Rápida
```bash
cd c:\Users\Boanerges\Desktop\Projetos\projeto-stgcn-cpraio
.\.venv\Scripts\python.exe src\predict_with_factions.py
```

**Output:**
- `outputs/predicoes_cvli.csv` - Scores por bairro
- `outputs/RELATORIO_PREDICOES.md` - Análise executiva
- `outputs/predicoes_cvli.json` - Estruturado para API

---

## 🔧 INSTALAÇÃO COMPLETA

### 1. Clonar/Setup Repositório
```bash
cd c:\Users\Boanerges\Desktop\Projetos
git clone <repo>
cd projeto-stgcn-cpraio
```

### 2. Environment Virtual
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Windows PowerShell: Se der erro de script
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3. Dependências
```bash
pip install -r requirements.txt
```

**Pacotes Principais:**
```
torch==2.x
numpy
pandas
geopandas
scikit-learn
```

### 4. Validar Instalação
```bash
python -c "import torch; print(torch.__version__)"
python src/model_faction_adapter.py  # Deve imprimir "✅ Modelo criado"
```

---

## 📁 ESTRUTURA DE PASTAS

```
projeto-stgcn-cpraio/
├── data/
│   ├── raw/                           # Dados brutos
│   ├── processed/                     # Tensores processados ⭐
│   │   ├── tensor_cvli_prisoes_faccoes.npy
│   │   ├── metadata_producao_v2.json
│   │   └── [outros tensores e csvs]
│   ├── graph/
│   │   ├── faccoes_23_01_2026/        # Snapshots de facções
│   │   │   ├── COMANDO VERMELHO.geojson
│   │   │   ├── MASSA.geojson
│   │   │   └── [5 facções mais]
│   │   └── [outros geojsons]
│   └── tensors/
│       └── dataset_producao_v2.pt
│
├── src/
│   ├── data/
│   │   ├── etl_producao_v2.py         ← ETL
│   │   ├── integrate_production_tensors.py
│   │   └── analyze_faction_movements.py
│   ├── model_faction_adapter.py       ← Modelo ⭐
│   ├── train_with_factions.py         ← Treino
│   ├── predict_with_factions.py       ← Predição ⭐
│   ├── config.py
│   ├── trainer.py
│   └── [outros módulos]
│
├── outputs/
│   ├── model_stgcn_faccoes.pth        ← Modelo treinado ⭐
│   ├── predicoes_cvli.csv             ← Forecasts ⭐
│   ├── RELATORIO_PREDICOES.md
│   └── [outros relatórios]
│
├── IMPLANTACAO_COMPLETA_FACCOES.md    ← Documentação
├── PRODUCAO_COM_FACCOES_SUMARIO.md
├── requirements.txt
└── README.md
```

---

## 🎯 CENÁRIOS DE USO

### Cenário 1: Previsão Diária
```bash
# Executar todo dia às 06:00
Schedule task ou cron job:
.\.venv\Scripts\python.exe src\predict_with_factions.py

# Envia output para:
# - outputs/predicoes_cvli.csv
# - outputs/RELATORIO_PREDICOES.md
# - Integrado em Dashboard/Email
```

### Cenário 2: Atualização Mensal de Facções
```bash
# Quando novo snapshot de facções disponível:

# 1. Adicionar dados
mkdir data/graph/faccoes_DD_MM_YYYY
# Colocar 7 GeoJSONs de facções

# 2. Reprocessar
.\.venv\Scripts\python.exe src\data\analyze_faction_movements.py

# 3. Re-treinar (opcional)
.\.venv\Scripts\python.exe src\train_with_factions.py

# 4. Novas predições
.\.venv\Scripts\python.exe src\predict_with_factions.py
```

### Cenário 3: Integração em API
```python
# Flask/FastAPI endpoint
from src.predict_with_factions import CVLIPredictor

predictor = CVLIPredictor(
    model_path='outputs/model_stgcn_faccoes.pth',
    tensor_path='data/processed/tensor_cvli_prisoes_faccoes.npy',
    metadata_path='data/processed/metadata_producao_v2.json'
)

@app.get("/api/forecast")
def forecast():
    predictions = predictor.predict_next_window()
    return predictions.to_dict('records')

@app.get("/api/risk/{bairro}")
def risk(bairro: str):
    predictions = predictor.predict_next_window()
    return predictions[predictions['bairro'] == bairro].to_dict()
```

### Cenário 4: Atualizar Dados de Entrada
```bash
# Se novos dados CVLI disponíveis:

# 1. Colocar em outputs/cvli_with_bairro.csv (tipo='cvli')
# 2. Colocar operacional em data/raw/View_Ocorrencias_Operacionais_Modelo.csv

# 3. Rodar ETL completo
.\.venv\Scripts\python.exe src\data\etl_producao_v2.py

# 4. Integrar tensores
.\.venv\Scripts\python.exe src\data\integrate_production_tensors.py

# 5. Análise de facções
.\.venv\Scripts\python.exe src\data\analyze_faction_movements.py

# 6. Retreinar
.\.venv\Scripts\python.exe src\train_with_factions.py

# 7. Predições
.\.venv\Scripts\python.exe src\predict_with_factions.py
```

---

## 🐳 DOCKER DEPLOYMENT (Opcional)

### Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]
```

### docker-compose.yml
```yaml
version: '3.8'
services:
  stgcn-predictor:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./data:/app/data
      - ./outputs:/app/outputs
    environment:
      - FLASK_ENV=production
```

### Build & Run
```bash
docker build -t stgcn-faccoes .
docker run -p 5000:5000 -v $(pwd)/data:/app/data stgcn-faccoes
```

---

## 📊 MONITORAMENTO E LOGS

### Setup de Logging
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
```

### Métricas para Acompanhar
```
1. Tempo de predição (ms)
2. Distribuição de CVLI por percentil
3. Bairros com risco alto (>75º)
4. Bairros com risco de mudança territorial (>30%)
5. Desvio vs. realizados (quando disponível)
```

### Health Check
```bash
# Validar modelo
python -c "
import torch
from src.model_faction_adapter import STGCN_DynamicFactions
model = STGCN_DynamicFactions()
X = torch.randn(1, 14, 121, 7)
output, aux = model(X, return_aux=True)
print('✓ Model OK')
"
```

---

## 🔐 SEGURANÇA

### Dados Sensíveis
- Modelo `outputs/model_stgcn_faccoes.pth`: Não publicar
- Snapshot de facções: Pode expor operações de inteligência
- Predições de CVLI: Restringir acesso

### Recomendações
```
1. Store model weights em S3/secure storage
2. API com autenticação OAuth2
3. Rate limiting: 100 req/min
4. Logging de queries (quem pediu o quê quando)
5. Criptografar dados em trânsito (HTTPS)
```

---

## 🐛 TROUBLESHOOTING

### Erro: `ModuleNotFoundError: No module named 'torch'`
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Erro: `'charmap' codec can't decode byte`
Usar encoding UTF-8:
```python
# Ao ler arquivos
with open(file, 'r', encoding='utf-8') as f:
    data = json.load(f)
```

### Erro: `CUDA out of memory`
Se usando GPU:
```python
torch.cuda.empty_cache()
# Ou usar CPU
device = torch.device('cpu')
```

### Predições todas zero
- Modelo ainda não treinado? Treinar com `train_with_factions.py`
- Tensor incorreto? Validar em `analyze_faction_movements.py`

### Modelo lento (CPU)
```python
# Considerar otimizações
model = torch.jit.script(model)  # TorchScript compilation
# Ou usar ONNX
import torch.onnx
torch.onnx.export(model, X, "model.onnx")
```

---

## 📈 PERFORMANCE TUNING

### Batch Size
```python
# Aumentar para mais throughput (se memória permitir)
batch_size = 32  # Default é 16
```

### Number of Workers
```python
dataloader = DataLoader(dataset, num_workers=4)  # Multiprocessing
```

### Mixed Precision (GPU)
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(X)
scaler.scale(loss).backward()
```

---

## 🔄 CI/CD PIPELINE

### GitHub Actions (`.github/workflows/predict.yml`)
```yaml
name: Daily Forecast

on:
  schedule:
    - cron: '0 6 * * *'  # 06:00 diariamente

jobs:
  forecast:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      
      - run: pip install -r requirements.txt
      - run: python src/predict_with_factions.py
      
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: predictions
          path: outputs/predicoes_cvli.csv
      
      - name: Slack notification
        run: |
          curl -X POST ${{ secrets.SLACK_WEBHOOK }} \
            -d '{"text":"Forecast Updated!"}'
```

---

## 📚 REFERÊNCIAS

### Arquivos Documentação
- `IMPLANTACAO_COMPLETA_FACCOES.md` - Visão geral completa
- `PRODUCAO_COM_FACCOES_SUMARIO.md` - Resumo técnico
- `data/processed/ADAPTACAO_MODELO_FACCOES.md` - Arquitetura neural
- `data/processed/RELATORIO_DINAMICA_FACCOES.md` - Análise de facções

### Scripts Principais
1. `src/data/etl_producao_v2.py` - Preparação de dados
2. `src/model_faction_adapter.py` - Modelo neural
3. `src/train_with_factions.py` - Treinamento
4. `src/predict_with_factions.py` - Inferência

---

## ✅ CHECKLIST DE DEPLOYMENT

- [ ] Ambiente virtual criado e ativado
- [ ] Dependências instaladas (`pip install -r requirements.txt`)
- [ ] Dados em lugar correto (`data/processed/tensor_cvli_prisoes_faccoes.npy`)
- [ ] Modelo treinado (`outputs/model_stgcn_faccoes.pth`)
- [ ] Teste rápido: `python src/predict_with_factions.py`
- [ ] API/cronjob configurado para execução regular
- [ ] Logs configurados
- [ ] Monitoramento ativado
- [ ] Backup de modelo (S3/cloud)
- [ ] Documentação compartilhada com team

---

## 📞 SUPORTE

**Problemas?**
1. Verificar logs: `logs/prediction.log`
2. Executar health check
3. Validar dados: `data/processed/metadata_producao_v2.json`
4. Consultar `IMPLANTACAO_COMPLETA_FACCOES.md`

**Atualizações?**
- Mensal: Novos snapshots de facções
- Trimestral: Retrainamento completo

---

**Pronto para Produção** ✅  
**Última Atualização:** 23/01/2026

