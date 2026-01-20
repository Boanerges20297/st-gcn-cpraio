# 🔧 TROUBLESHOOTING: Limites do Ceará no Mapa

**Status:** ✅ Testes implementados  
**Data:** 20/01/2026

---

## 📋 AÇÕES REALIZADAS

### 1️⃣ Script de Download do IBGE
**Arquivo:** `scripts_ajuste/15_buscar_limites_ibge.py`

Criado script que baixa limites **direto do IBGE** em 3 formatos:

```
data/raw/
├── limites_ceara_ibge_linhas.geojson    (2.2 KB) - LineString - PREFERIDO
├── limites_ceara_ibge_simples.geojson   (2.5 KB) - GeoJSON simples
└── limites_ceara_ibge_completo.geojson  (2.5 KB) - GeoJSON completo
```

**Vantagem:** Dados mais limpos do IBGE + múltiplas opções de teste

### 2️⃣ Página de Teste Isolada
**Arquivo:** `src/templates/teste_limites.html`  
**Rota:** `http://localhost:5000/teste-limites`

Página HTML simples que testa:
- ✅ Carregamento dos 3 arquivos (fallback automático)
- ✅ Renderização com linha **vermelha 1pt** (bem fina, sem opacidade)
- ✅ Auto-zoom para enquadrar o Ceará
- ✅ Debug logs no console

### 3️⃣ Atualização do Dashboard
**Arquivo:** `src/templates/dashboard_estrategico.html`

Mudanças:
- ✅ Novo script que usa LineString do IBGE primeiro
- ✅ Fallback automático para arquivo original
- ✅ Função helper `carregarLimitesNoMapa()` reutilizável
- ✅ Estilo bem simples: **vermelho 1pt, sem opacidade**

### 4️⃣ Novo Endpoint de Teste
**Arquivo:** `src/app.py`

Adicionado:
- `/teste-limites` - página de teste isolada
- `/api/test_geojson` - diagnóstico do servidor

---

## 🧪 FLUXO DE TESTE

### Teste 1: Página Isolada (Mais fácil de debugar)
```
1. Acesse: http://localhost:5000/teste-limites
2. Abra console (F12)
3. Procure por:
   - ✓ "Sucesso na tentativa X"
   - Linha VERMELHA no mapa
```

**Se aparecer aqui:** Problema está no dashboard  
**Se não aparecer:** Problema está na renderização Leaflet/servidor

### Teste 2: Verificar Endpoint
```
http://localhost:5000/api/test_geojson
```

Deve retornar JSON com:
```json
{
  "status": "ok",
  "features": 1,
  "geometry_types": ["Polygon"],
  ...
}
```

### Teste 3: Dashboard Completo
```
1. http://localhost:5000/dashboard-estrategico
2. Abra "Facções em Atuação & Geolocalização"
3. Procure por linha VERMELHA com Ceará
4. Abra console (F12) para logs
```

---

## 📊 COMPARATIVO DOS 3 ARQUIVOS

| Arquivo | Tamanho | Tipo | Vantagem |
|---------|---------|------|----------|
| `limites_ceara_ibge_linhas.geojson` | 2.2 KB | LineString | Mais leve, sem fill |
| `limites_ceara_ibge_simples.geojson` | 2.5 KB | Polygon | Padrão, sem propriedades extras |
| `limites_ceara.geojson` | 15.6 KB | Polygon | Original, mais completo |

**Estratégia:** Tenta linhas → simples → completo

---

## 🎨 ESTILO TESTADO

```javascript
style: {
    color: '#FF0000',           // Vermelho bem visível
    weight: 1,                  // 1px (bem fino)
    opacity: 1,                 // Sem transparência
    fill: false,                // Sem preenchimento
    fillOpacity: 0
}
```

**Razão:** Linha vermelha fininha é mais fácil de ver se renderiza ou não

---

## 📝 INSTRUÇÕES FINAIS

### ✅ Se a linha VERMELHA aparecer no teste isolado:

1. O problema está na função `carregarMapaFaccoes()` do dashboard
2. Copie a função `carregarLimitesNoMapa()` do código do teste para lá
3. Chame: `carregarLimitesNoMapa(ceara, mapaFaccoes);`

### ❌ Se a linha NÃO aparecer em nenhum teste:

1. Verifique console (F12) - deve haver erro específico
2. Teste: `fetch('/data/raw/limites_ceara_ibge_linhas.geojson')`
3. Se falhar → problema no servidor
4. Se OK mas não renderiza → problema do Leaflet CSS/JS

### 🔍 Debug via Console

```javascript
// No console do navegador (F12), teste:
fetch('/data/raw/limites_ceara_ibge_linhas.geojson')
  .then(r => r.json())
  .then(d => console.log('GeoJSON:', d))
  .catch(e => console.error('Erro:', e));
```

---

## 🚀 PRÓXIMOS PASSOS

1. **Executar teste isolado** → Confirmar que Leaflet funciona
2. **Se OK:** Copiar função para dashboard
3. **Se falhar:** Investigar erro específico no console
4. **Se múltiplas linhas aparecerem:** Adicionar `.clearLayers()` antes de adicionar

---

## 📌 RESUMO TÉCNICO

- ✅ 3 arquivos GeoJSON criados (IBGE)
- ✅ Página de teste isolada criada
- ✅ Fallback automático implementado
- ✅ Estilo super simples (1pt, sem opacidade)
- ✅ Debug logs adicionados
- ✅ Rota de teste criada

**Status:** Pronto para testar!

