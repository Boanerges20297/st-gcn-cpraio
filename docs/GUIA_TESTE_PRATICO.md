# 🧪 GUIA DE TESTE PRÁTICO

## ⚡ Quick Start

### 1. Iniciar o servidor
```bash
cd c:\Users\Boanerges\Desktop\Projetos\projeto-stgcn-cpraio
.venv\Scripts\python.exe src/app.py
```

Você deve ver:
```
WARNING in app.run():
  Use a production WSGI server instead.
Running on http://127.0.0.1:5000
```

### 2. Abrir no navegador
```
http://localhost:5000/
```

---

## 📋 CHECKLIST DE TESTES

### Teste 1: Mapa Tático Funciona
```
□ Abrir http://localhost:5000/
□ Ver "SIGERAIO" no título
□ Mapa com clusters de crimes aparecer
□ Sidebar com filtros à esquerda
□ Novo botão [🤖 Dashboard] visível
```

### Teste 2: Clicar no Botão Dashboard
```
□ Encontrar botão [🤖 Dashboard] no topo do sidebar
□ Clicar nele
□ Ser redirecionado para /dashboard-estrategico
□ Página carrega sem erros
```

### Teste 3: Dashboard Descritivo Funciona
```
□ Ver título "🎯 Dashboard Estratégico"
□ Ver números de crimes (83.295 total)
□ Ver distribuição por facção (CV, TCP)
□ Ver top 10 bairros com cores de alerta
□ Ver botão [← Voltar] no topo direito
```

### Teste 4: Gerar Análise com IA
```
□ Clicar em [⚡ Gerar Análise]
□ Ver spinner loading
□ Aguardar 5-30 segundos
□ Ver parecer da IA aparecer
□ Parecer tem recomendações de bairros
```

### Teste 5: Voltar ao Mapa
```
□ Clicar em [← Voltar] no dashboard
□ Ser redirecionado para http://localhost:5000/
□ Mapa aparecer normalmente
```

### Teste 6: Navegação Bidirecional
```
□ Mapa → [🤖 Dashboard] → Dashboard funciona
□ Dashboard → [← Voltar] → Mapa funciona
□ Repetir 3x (deve funcionar sempre)
```

---

## 🎯 TESTE DE COMPARAÇÃO (Principal)

### Abra em Duas Abas

**Aba 1**: http://localhost:5000/
**Aba 2**: http://localhost:5000/dashboard-estrategico

Alt+Tab entre elas para comparar:

```
ABA 1 (Mapa)
├─ [🤖 Dashboard] botão
├─ Região: [CAPITAL ▼]
├─ Facção: [TODAS ▼]
├─ Tipo Crime: [TODOS ▼]
├─ Top 5 Alvos
└─ Mapa com clusters

ABA 2 (Dashboard)
├─ [← Voltar] botão
├─ 📊 Números consolidados
├─ 👥 Distribuição facções
├─ 🚨 138 bairros com risco
└─ 🤖 [⚡ Gerar Análise]
```

### Perguntas para Responder

1. **Qual é mais intuitivo para UM GESTOR?**
   - [ ] Mapa (vê geograficamente)
   - [ ] Dashboard (números + IA)
   - [ ] Os dois complementam

2. **Qual é mais rápido para DECISÃO?**
   - [ ] Mapa (filtrar + olhar)
   - [ ] Dashboard (ler parecer IA)

3. **Qual REDUZ TEMPO de análise?**
   - [ ] Mapa (precisa interpretar clusters)
   - [ ] Dashboard (IA já recomenda)

4. **Como integrar SIGERAIO?**
   - [ ] Manter só o mapa
   - [ ] Manter só o dashboard
   - [ ] Os dois lado a lado
   - [ ] Abas/tabs intercambiáveis

---

## 🔧 TROUBLESHOOTING

### Problema: Servidor não inicia
```
Erro: "Address already in use"
→ Solução: Matar processo anterior
   taskkill /F /IM python.exe
   Depois: .venv\Scripts\python.exe src/app.py
```

### Problema: Mapa não carrega
```
Erro: "Leaflet não encontrado"
→ Solução: Verificar internet (precisa de CDN)
   Se offline, rodar com Chrome → F12 → Offline
```

### Problema: Dashboard mostra "Dados não disponíveis"
```
Erro: "CONSOLIDATED_FILE not found"
→ Solução: Rodar ETL primeiro
   .venv\Scripts\python.exe src/etl.py
```

### Problema: "Análise gerou erro"
```
Erro: "403 Your API key was reported as leaked"
→ Solução: Atualizar GEMINI_KEY no .env
   Ou usar chave diferente (GEMINI_KEY_2, GEMINI_KEY_3)
```

### Problema: Análise muito lenta (>30s)
```
Gemini pode estar saturado
→ Solução: Esperar e tentar novamente
   Ou usar chave diferente
```

---

## 📊 DADOS ESPERADOS

### Mapa Tático
```
Total crimes: 83.295
├─ CAPITAL: 55.252 (66%)
├─ RMF: 21.665 (26%)
└─ INTERIOR: 6.378 (8%)

Bairros em Fortaleza: 138
Top críticos: DE LOURDES, AUTRAN NUNES, VICENTE PINZÓN
```

### Dashboard Estratégico
```
Total crimes consolidados: 83.295
Crimes em CAPITAL: 55.252
CVP (roubos): 69.046
CVLI (homicídios): 14.249

Facções:
├─ CV: 55.251
└─ TCP: 1

Predição de risco (15 dias):
├─ DE LOURDES: 33.3% 🔴
├─ AUTRAN NUNES: 32.5% 🔴
└─ ... 136 outros bairros
```

---

## 🎬 ROTEIRO COMPLETO DE TESTE

### Minuto 0-2: Setup
```
1. Abrir terminal PowerShell
2. cd c:\Users\Boanerges\Desktop\Projetos\projeto-stgcn-cpraio
3. .venv\Scripts\python.exe src/app.py
4. Aguardar "Running on http://127.0.0.1:5000"
```

### Minuto 2-3: Abrir Navegador
```
5. Ctrl+T (nova aba)
6. http://localhost:5000/
7. Aguardar mapa carregar (5-10s)
8. Ver [🤖 Dashboard] no topo
```

### Minuto 3-5: Testar Mapa
```
9. Selecionar Região = CAPITAL
10. Selecionar Facção = CV
11. Ver mapa territorial atualizar
12. Observar Top 5 alvos
```

### Minuto 5-6: Ir para Dashboard
```
13. Clicar [🤖 Dashboard]
14. Aguardar página carregar
15. Ver números de crimes
16. Ver distribuição de facções
```

### Minuto 6-15: Análise com IA
```
17. Clicar [⚡ Gerar Análise]
18. Spinner aparecer
19. Aguardar 5-30 segundos
20. Ver parecer da IA
21. Ler recomendações
```

### Minuto 15-16: Voltar e Comparar
```
22. Clicar [← Voltar]
23. Voltar ao mapa
24. Alt+Tab entre abas
25. Comparar qual é mais útil
```

### Minuto 16+: Feedback
```
26. Decidir qual abordagem usar
27. Documentar preferência
28. Sugerir ajustes
```

---

## ✅ SUCESSO!

Se você conseguiu:

- ✓ Abrir o mapa
- ✓ Clicar no botão Dashboard
- ✓ Ver o dashboard descritivo
- ✓ Clicar em "Gerar Análise"
- ✓ Receber parecer da IA
- ✓ Voltar ao mapa
- ✓ Navegar nos dois caminhos

**Então a integração foi bem-sucedida!** 🎉

---

## 💬 FEEDBACK

Após testar, responda:

1. **Qual você usaria no dia a dia?**
   - [ ] Só mapa
   - [ ] Só dashboard
   - [ ] Os dois

2. **O que melhorar no dashboard?**
   - [ ] Adicionar mais gráficos
   - [ ] Adicionar mais números
   - [ ] Mudar layout
   - [ ] Tudo bem

3. **A IA ajudou na decisão?**
   - [ ] Sim, muito claro
   - [ ] Parcialmente
   - [ ] Não entendi

4. **Integração com SIGERAIO OK?**
   - [ ] Sim, botão funciona
   - [ ] Precisa melhorar
   - [ ] Não gostei

---

**Pronto para começar?** 🚀

Rode: `.venv\Scripts\python.exe src/app.py`

E acesse: `http://localhost:5000/`
