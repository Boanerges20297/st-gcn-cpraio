# GUIA TÉCNICO: IMPLEMENTAÇÃO DOS NOVOS PARÂMETROS

## 🔧 Referência Rápida - Mudanças no Código

---

## 1. CONFIG.PY - Adicionar Definições CVLI

### Antes:
```python
# Apenas hiperparâmetros
HyperParams = {
    'window_size': 14,
    'target_window': 15,
    'hidden_dim': 32,
}
```

### Depois:
```python
# Adicionar definições de CVLI
CVLI_DEFINITIONS = {
    'homicidio': ['HOMICÍDIO', 'TENTATIVA DE HOMICIDIO', 'MORTE'],
    'estupro': ['ESTUPRO'],
    'roubo_violento': ['ROUBO', 'ROUBO DE VEÍCULO'],
    'lesao_corporal': ['LESÃO CORPORAL'],
}

CVLI_KEYWORDS = []
for category, keywords in CVLI_DEFINITIONS.items():
    CVLI_KEYWORDS.extend(keywords)

# Pesos de prioridade
CRIME_WEIGHTS = {
    'is_cvli': 3.0,                    # CVLI = 3x peso
    'drug_seizure_1kg': 2.0,           # Grande apreensão = 2x
    'weapons_and_drugs': 2.0,          # Arma+droga = 2x
    'arrest_in_territory': 1.5,        # Prisão local = 1.5x
}

# Dimensionalidade do novo tensor
X_FEATURE_DIMENSIONS = {
    'total_crimes': 1,                 # Contagem base
    'cvli_crimes': 1,                  # Apenas CVLI
    'drug_seizures_g': 1,              # Drogas em gramas
    'weapons_count': 1,                # Quantidade de armas
    'arrest_count': 1,                 # Presos por dia
    'drug_value_estimate': 1,          # Valor estimado drogas
    'territory_stability': 1,          # Score 0-1
}
X_TOTAL_FEATURES = sum(X_FEATURE_DIMENSIONS.values())  # = 7
```

---

## 2. DATA_LOADER.PY - Normalização de Dados

### Adicionar funções para CVLI e limpar JSON operacional:

```python
def detect_cvli(natureza_str, cvli_keywords=config.CVLI_KEYWORDS):
    """
    Identifica se uma ocorrência é CVLI baseado em palavras-chave.
    
    Args:
        natureza_str: String da natureza da ocorrência
        cvli_keywords: Lista de palavras-chave CVLI
    
    Returns:
        bool: True se é CVLI
    """
    if not isinstance(natureza_str, str):
        return False
    
    natureza_upper = natureza_str.upper()
    return any(kw in natureza_upper for kw in cvli_keywords)


def extract_lat_long(lat_long_str):
    """
    Extrai latitude e longitude do campo 'lat_long' do JSON operacional.
    
    Formato esperado: "-3.7668038,-38.584197"
    
    Args:
        lat_long_str: String com coords separadas por vírgula
    
    Returns:
        tuple: (lat, long) como floats, ou (None, None) se inválido
    """
    if not isinstance(lat_long_str, str):
        return None, None
    
    try:
        parts = lat_long_str.strip().split(',')
        lat = float(parts[0].strip())
        long = float(parts[1].strip())
        return lat, long
    except (ValueError, IndexError):
        return None, None


def parse_operational_json(json_path):
    """
    Converte JSON operacional para DataFrame estruturado.
    
    Args:
        json_path: Caminho para ocorrencia_policial_operacional.json
    
    Returns:
        pd.DataFrame: Dados normalizados
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Filtrar apenas registros de dados (tipo='table' com data)
    records = []
    for item in raw_data:
        if item.get('type') == 'table' and 'data' in item:
            records.extend(item['data'])
    
    df = pd.DataFrame(records)
    
    # Limpeza de colunas
    df = df.rename(columns={
        'Data': 'data',
        'CidadeOcor': 'municipio',
        'BairroOcor': 'bairro',
        'Natureza': 'natureza',
        'total_armas_cache': 'total_armas',
        'total_drogas_cache': 'total_drogas_g',
        'Dinheiro_Apreendido': 'dinheiro_apreendido',
        'area_faccao': 'area_faccao',
        'LocalOcor': 'local_ocorrencia',
    })
    
    # Converter tipos
    df['data'] = pd.to_datetime(df['data'], format='%Y-%m-%d', errors='coerce')
    df['total_armas'] = pd.to_numeric(df['total_armas'], errors='coerce').fillna(0).astype(int)
    df['total_drogas_g'] = pd.to_numeric(df['total_drogas_g'], errors='coerce').fillna(0)
    df['dinheiro_apreendido'] = pd.to_numeric(df['dinheiro_apreendido'], errors='coerce').fillna(0)
    
    # Extrair lat/long
    coords = df['lat_long'].apply(extract_lat_long)
    df['lat'] = coords.apply(lambda x: x[0])
    df['long'] = coords.apply(lambda x: x[1])
    
    # Adicionar features engineerizadas
    df['is_cvli'] = df['natureza'].apply(detect_cvli)
    df['has_large_seizure'] = df['total_drogas_g'] >= 1000
    df['has_weapons_drugs'] = (df['total_armas'] > 0) & (df['total_drogas_g'] > 0)
    
    # Descartar registros sem geolocalização
    df = df.dropna(subset=['lat', 'long'])
    df = df[df['data'].notna()]
    
    return df


def parse_tropa_narrative(narrative, base_fields):
    """
    Extrai informações estruturadas de narrativa de tropa.
    
    Args:
        narrative: Texto narrativo da ocorrência de tropa
        base_fields: Dict com campos básicos (data, municipio, etc)
    
    Returns:
        dict: Informações estruturadas extraídas
    """
    import re
    
    result = dict(base_fields)
    
    if not isinstance(narrative, str):
        return result
    
    narrative_upper = narrative.upper()
    
    # Detectar tipo de operação
    if 'MORTE' in narrative_upper or 'ÓBITO' in narrative_upper:
        result['tipo_operacao'] = 'morte'
    elif 'APREENSÃO' in narrative_upper or 'APREENDIDO' in narrative_upper:
        result['tipo_operacao'] = 'apreensao'
    elif 'PRISÃO' in narrative_upper or 'PRESO' in narrative_upper:
        result['tipo_operacao'] = 'prisao'
    
    # Extrair quantidade de presos (padrão: "1 preso", "2 presos", "voz de prisão")
    match_presos = re.search(r'(\d+)\s+preso', narrative_upper)
    result['num_presos'] = int(match_presos.group(1)) if match_presos else 0
    
    # Contar menções de "arma", "droga", "tráfico"
    result['arma_mencionada'] = 'ARMA' in narrative_upper
    result['droga_mencionada'] = 'DROGA' in narrative_upper or 'ENTORPECENTE' in narrative_upper
    
    # Extrair valores (padrão: "R$ 1.000", "1000G", etc)
    match_valor = re.search(r'R\$\s*([\d.,]+)', narrative)
    if match_valor:
        result['valor_apreendido'] = float(match_valor.group(1).replace(',', ''))
    else:
        result['valor_apreendido'] = 0
    
    return result


def normalize_tropa_dataset(tropa_json_path):
    """
    Normaliza ocorrências_tropa.json para DataFrame.
    
    Args:
        tropa_json_path: Caminho para ocorrencias_tropa.json
    
    Returns:
        pd.DataFrame: Dados de prisões normalizados
    """
    with open(tropa_json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    records = []
    for item in raw_data:
        if isinstance(item, dict) and 'ocorrencia' in item:
            # Extrair campos básicos
            base = {
                'id_ocorrencia': item.get('id_ocorrencia'),
                'base_origem': item.get('base_origem'),
                'data': item.get('data_registro', '').split()[0],  # YYYY-MM-DD
                'municipio': None,  # Extrair da narrativa se possível
                'narrativa_original': item.get('ocorrencia', ''),
            }
            
            # Fazer parsing da narrativa
            parsed = parse_tropa_narrative(item.get('ocorrencia', ''), base)
            records.append(parsed)
    
    df = pd.DataFrame(records)
    
    # Converter tipos
    df['data'] = pd.to_datetime(df['data'], format='%Y-%m-%d', errors='coerce')
    
    return df.dropna(subset=['data'])
```

---

## 3. GRAPH_BUILDER.PY - Integrar Novas Features

### Modificar build_graph() para usar edge_weights:

```python
def build_graph_with_weights(region_name, df_full):
    """
    Constrói grafo com topologia física + lógica + pesos baseados em crimes.
    
    Args:
        region_name: 'CAPITAL', 'RMF', ou 'INTERIOR'
        df_full: DataFrame unificado com crimes
    
    Returns:
        dict: {edge_index, edge_weight, X_extended, nodes}
    """
    print(f"\n>>> CONSTRUINDO GRAFO COM PESOS: {region_name}")
    
    # 1. Carregar mapa
    geo_path = config.GEOJSON_PATHS[region_name]
    gdf = gpd.read_file(geo_path)
    nodes = gdf['name'].str.upper().str.strip().tolist()
    node_map = {name: i for i, name in enumerate(nodes)}
    
    df_region = df_full[df_full['regiao_sistema'] == region_name].copy()
    
    # 2. Construir adjacency list com pesos
    edges = []
    edge_weights = []
    
    # A. Vizinhos físicos (base)
    for idx, row in tqdm(gdf.iterrows(), desc="Topologia Física"):
        neighbors = gdf[gdf.geometry.touches(row.geometry)].index.tolist()
        node_name = nodes[idx]
        
        for n_idx in neighbors:
            edges.append([idx, n_idx])
            
            # Peso base para vizinhos físicos = 1.0
            weight = 1.0
            
            # Aumentar peso se há CVLI nessa região
            cvli_count = len(df_region[
                (df_region['local_oficial'] == node_name) & 
                (df_region['is_cvli'] == True)
            ])
            if cvli_count > 0:
                weight *= config.CRIME_WEIGHTS['is_cvli']
            
            edge_weights.append(weight)
    
    # B. Conexões lógicas (facções)
    node_factions = {}
    for node_name in nodes:
        factions = df_region[df_region['local_oficial'] == node_name]['area_faccao']
        if not factions.empty:
            node_factions[node_name] = factions.mode()[0] if len(factions.mode()) > 0 else 'DESCONHECIDO'
    
    faction_groups = {}
    for name, fac in node_factions.items():
        if fac not in ['DESCONHECIDO', 'NEUTRO', 'nan', 'SEM_FACCAO']:
            if fac not in faction_groups:
                faction_groups[fac] = []
            if name in node_map:
                faction_groups[fac].append(node_map[name])
    
    for fac, indices in faction_groups.items():
        if len(indices) > 1:
            for i in indices:
                targets = random.sample(indices, min(len(indices), 3))
                for t in targets:
                    if i != t:
                        edges.append([i, t])
                        
                        # Peso para conexão faccionada
                        weight = 1.5  # Base para lógica
                        
                        # Aumentar se há apreensões grandes ou arma+droga
                        node_name = nodes[i]
                        large_seizures = len(df_region[
                            (df_region['local_oficial'] == node_name) & 
                            (df_region['has_large_seizure'] == True)
                        ])
                        weapons_drugs = len(df_region[
                            (df_region['local_oficial'] == node_name) & 
                            (df_region['has_weapons_drugs'] == True)
                        ])
                        
                        if large_seizures > 0:
                            weight *= config.CRIME_WEIGHTS['drug_seizure_1kg']
                        if weapons_drugs > 0:
                            weight *= config.CRIME_WEIGHTS['weapons_and_drugs']
                        
                        edge_weights.append(weight)
    
    # 3. Converter para tensor
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weights, dtype=torch.float).view(-1)
    
    # 4. Construir tensor X estendido (num_days, num_nodes, K features)
    df_region['data'] = pd.to_datetime(df_region['data'])
    min_date = df_region['data'].min()
    max_date = df_region['data'].max()
    all_dates = pd.date_range(min_date, max_date, freq='D')
    
    num_days = len(all_dates)
    num_nodes = len(nodes)
    K = config.X_TOTAL_FEATURES
    
    X_extended = torch.zeros((num_days, num_nodes, K), dtype=torch.float)
    date_map = {d.date(): i for i, d in enumerate(all_dates)}
    
    daily_agg = df_region.groupby([df_region['data'].dt.date, 'local_oficial']).agg({
        'id': 'count',                  # Total crimes
        'is_cvli': 'sum',               # CVLI count
        'total_drogas_g': 'sum',        # Total drogas
        'total_armas': 'sum',           # Total armas
        'num_presos': 'sum',            # Total presos
        'dinheiro_apreendido': 'sum',   # Valor confiscado
    }).reset_index()
    daily_agg.columns = ['data', 'local_oficial', 'total_crimes', 'cvli_crimes', 
                         'total_drogas_g', 'total_armas', 'num_presos', 'dinheiro_apreendido']
    
    for _, row in daily_agg.iterrows():
        d_idx = date_map.get(row['data'])
        n_idx = node_map.get(row['local_oficial'])
        
        if d_idx is not None and n_idx is not None:
            feature_idx = 0
            X_extended[d_idx, n_idx, feature_idx] = row['total_crimes']
            feature_idx += 1
            X_extended[d_idx, n_idx, feature_idx] = row['cvli_crimes']
            feature_idx += 1
            X_extended[d_idx, n_idx, feature_idx] = row['total_drogas_g']
            feature_idx += 1
            X_extended[d_idx, n_idx, feature_idx] = row['total_armas']
            feature_idx += 1
            X_extended[d_idx, n_idx, feature_idx] = row['num_presos']
            feature_idx += 1
            X_extended[d_idx, n_idx, feature_idx] = row['dinheiro_apreendido'] / 100  # Normalizar
            feature_idx += 1
            
            # Territory stability score (0-1): normalized crime reduction over time
            X_extended[d_idx, n_idx, feature_idx] = 0.5  # Placeholder
    
    print(f"    - Tensor X: {X_extended.shape}")
    print(f"    - Edge weights: min={edge_weight.min():.2f}, max={edge_weight.max():.2f}")
    
    return {
        'edge_index': edge_index,
        'edge_weight': edge_weight,
        'X': X_extended,
        'nodes': nodes,
        'date_range': all_dates
    }
```

---

## 4. MODEL.PY - Usar Edge Weights

### Modificar GCN para usar pesos:

```python
class STGCN_Cpraio_v2(nn.Module):
    """
    Versão melhorada com suporte a edge weights e features estendidas.
    """
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels, dropout=0.3):
        super(STGCN_Cpraio_v2, self).__init__()
        
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.dropout_rate = dropout
        
        # LSTM: Processa série temporal
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_channels,
            batch_first=True,
            dropout=dropout if hidden_channels > 1 else 0
        )
        
        # GCN: Com suporte a edge weights
        self.gcn1 = GCNConv(hidden_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        
        # Head: Predição
        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, out_channels)
    
    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: (batch, window_size, num_nodes, in_channels)
            edge_index: (2, num_edges)
            edge_weight: (num_edges,) [NOVO]
        """
        batch_size, window_size, num_nodes, num_features = x.size()
        
        # 1. LSTM temporal
        x_lstm = x.view(batch_size * num_nodes, window_size, num_features)
        _, (h_n, _) = self.lstm(x_lstm)
        x_spatial = h_n.squeeze(0)  # (batch*nodes, hidden)
        
        # 2. GCN com edge_weight
        x_gcn = self.gcn1(x_spatial, edge_index, edge_weight)
        x_gcn = F.relu(x_gcn)
        x_gcn = F.dropout(x_gcn, p=self.dropout_rate, training=self.training)
        
        x_gcn = self.gcn2(x_gcn, edge_index, edge_weight)
        x_gcn = F.relu(x_gcn)
        x_gcn = F.dropout(x_gcn, p=self.dropout_rate, training=self.training)
        
        # 3. Head predição
        x_out = self.fc1(x_gcn)
        x_out = F.relu(x_out)
        out = self.fc2(x_out)
        
        out = out.view(batch_size, num_nodes, -1)
        return out
```

---

## 5. TRAINER.PY - Atualizar com Novos Dados

### Adaptar train_region() para usar dados estruturados:

```python
def train_region_v2(region_name, df_unified, use_edge_weights=True):
    """
    Versão melhorada do treino com dados unificados.
    
    Args:
        region_name: 'CAPITAL', 'RMF', 'INTERIOR'
        df_unified: DataFrame com todas as features
        use_edge_weights: Boolean para ativar pesos nas edges
    """
    print(f"\n{'='*50}")
    print(f" [TREINADOR v2] Iniciando: {region_name}")
    print(f"{'='*50}")
    
    # 1. Construir grafo COM PESOS
    graph_data = build_graph_with_weights(region_name, df_unified)
    
    X_full = graph_data['X']
    edge_index = graph_data['edge_index']
    edge_weight = graph_data['edge_weight'] if use_edge_weights else None
    
    # 2. Validação
    min_days = config.HyperParams['window_size'] + config.HyperParams['target_window'] + 5
    if len(X_full) < min_days:
        print(f"[!] Histórico muito curto ({len(X_full)} dias). Mínimo: {min_days}.")
        return
    
    # 3. Normalização por feature
    X_norm = X_full.clone()
    stats = {}
    for feat_idx in range(X_full.shape[2]):
        mean = X_full[:, :, feat_idx].mean()
        std = X_full[:, :, feat_idx].std() + 1e-5
        X_norm[:, :, feat_idx] = (X_full[:, :, feat_idx] - mean) / std
        stats[feat_idx] = {'mean': mean.item(), 'std': std.item()}
    
    # Salvar stats
    torch.save(stats, config.ARTIFACTS[region_name]['stats'])
    
    # 4. Dataset e DataLoader
    dataset = CrimeSeriesDataset(X_norm, window_size=14, target_window=15)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # 5. Modelo v2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = STGCN_Cpraio_v2(
        num_nodes=X_full.shape[1],
        in_channels=X_full.shape[2],  # K features
        hidden_channels=32,
        out_channels=1  # Prever próximos 15 dias
    ).to(device)
    
    # 6. Otimizador com pesos decrescentes
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    # 7. Loop de treinamento
    best_val_loss = float('inf')
    patience = 25
    patience_counter = 0
    
    for epoch in range(250):  # Aumentado de 200
        # TRAIN
        model.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            edge_index_dev = edge_index.to(device)
            edge_weight_dev = edge_weight.to(device) if edge_weight is not None else None
            
            optimizer.zero_grad()
            y_pred = model(x_batch, edge_index_dev, edge_weight_dev)
            
            # Prever apenas primeira feature (total_crimes)
            y_pred = y_pred[:, :, 0].mean(dim=1)  # Average across nodes
            y_true = y_batch[:, :, 0].mean(dim=1)  # Target também média
            
            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # VALIDATION
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                edge_index_dev = edge_index.to(device)
                edge_weight_dev = edge_weight.to(device) if edge_weight is not None else None
                
                y_pred = model(x_batch, edge_index_dev, edge_weight_dev)
                y_pred = y_pred[:, :, 0].mean(dim=1)
                y_true = y_batch[:, :, 0].mean(dim=1)
                
                loss = criterion(y_pred, y_true)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/250 | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), config.ARTIFACTS[region_name]['model'])
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[Early Stopping] Epoch {epoch+1}")
                break
    
    print(f"[✓] Modelo salvo: {config.ARTIFACTS[region_name]['model']}")
    return model

```

---

## 6. TESTE DE INTEGRAÇÃO

### Script para validar pipeline:

```python
# test_new_model_pipeline.py
import sys
sys.path.insert(0, '/path/to/st-gcn_cpraio/src')

from data_loader import parse_operational_json, normalize_tropa_dataset
import pandas as pd

# 1. Carregar dados operacionais
print("1. Carregando dados operacionais...")
df_op = parse_operational_json('data/raw/ocorrencia_policial_operacional.json')
print(f"   ✓ {len(df_op)} registros, {df_op['is_cvli'].sum()} CVLI")
print(f"   ✓ {df_op['has_large_seizure'].sum()} com apreensão >= 1kg")
print(f"   ✓ {df_op['has_weapons_drugs'].sum()} com arma + droga")

# 2. Carregar dados de tropa
print("\n2. Carregando dados de prisões (tropa)...")
df_tropa = normalize_tropa_dataset('data/raw/ocorrencias_tropa.json')
print(f"   ✓ {len(df_tropa)} registros")
print(f"   ✓ {df_tropa['num_presos'].sum()} total de presos")

# 3. Unificar
print("\n3. Unificando datasets...")
df_unified = pd.concat([df_op, df_tropa], ignore_index=True)
df_unified = df_unified.sort_values('data').reset_index(drop=True)
print(f"   ✓ {len(df_unified)} registros unificados")

# 4. Testar construção de grafo
print("\n4. Construindo grafo com pesos...")
from graph_builder import build_graph_with_weights
graph = build_graph_with_weights('CAPITAL', df_unified)
print(f"   ✓ Edge index shape: {graph['edge_index'].shape}")
print(f"   ✓ Edge weights: {graph['edge_weight'].shape}")
print(f"   ✓ X tensor: {graph['X'].shape}")

print("\n✅ Pipeline validado com sucesso!")
```

---

## 📋 Checklist de Implementação

- [ ] Atualizar `config.py` com CVLI_DEFINITIONS
- [ ] Implementar `extract_lat_long()` em `data_loader.py`
- [ ] Implementar `detect_cvli()` em `data_loader.py`
- [ ] Implementar `parse_operational_json()` em `data_loader.py`
- [ ] Implementar `parse_tropa_narrative()` em `data_loader.py`
- [ ] Implementar `normalize_tropa_dataset()` em `data_loader.py`
- [ ] Criar `build_graph_with_weights()` em `graph_builder.py`
- [ ] Criar `STGCN_Cpraio_v2` em `model.py`
- [ ] Criar `train_region_v2()` em `trainer.py`
- [ ] Testar com `test_new_model_pipeline.py`
- [ ] Executar treinamento completo
- [ ] Validar backtest 2025

---

**Próximo passo:** Começar com Task 1.1 (normalização JSON operacional)
