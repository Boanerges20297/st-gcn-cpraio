# 🔧 GUIA DE CORREÇÃO - CHAVES API NÃO FUNCIONANDO

## 📊 Status Atual

```
✓ Chaves presentes: 3/3
✓ Formato válido: 3/3
✗ Funcionais: 0/3 ❌

CHAVE 1: Permissão Negada (API não ativada)
CHAVE 2: Cota Excedida (Free tier saturado)
CHAVE 3: Cota Excedida (Free tier saturado)
```

---

## 🎯 Problema Identificado

Você criou as chaves **sem ativar** a "Generative Language API" no projeto Google Cloud.

Além disso, as chaves 2 e 3 já ultrapassaram a cota free tier (15 requisições/minuto).

---

## ⚡ Solução em 3 Passos

### PASSO 1: Ativar a API (CRÍTICO - 5 minutos)

1. Abra: https://console.cloud.google.com/
2. Faça login com sua conta Google
3. Selecione o **projeto 288580115108** (ou o projeto correto)
4. Vá para **"APIs & Services"** > **"Library"**
5. Procure por **"Generative Language API"**
6. Clique no resultado
7. Clique em **"ENABLE"** (azul)
8. Aguarde 2-3 minutos para propagação

**Link direto:**
https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com

---

### PASSO 2: Adicionar Billing (RECOMENDADO - 5 minutos)

Para evitar limitações de quota, configure um método de pagamento:

1. Acesse: https://console.cloud.google.com/
2. Vá para **"Billing"** > **"Overview"**
3. Clique em **"Link Billing Account"**
4. Siga as instruções para adicionar seu cartão
5. **Google oferece $300 de crédito gratuito** para novos usuários
6. Com billing ativado, sua quota aumenta para:
   - ✓ 2 requisições/segundo (vs 15/minuto no free tier)
   - ✓ 1 milhão de tokens/minuto (vs muito limitado)

---

### PASSO 3: Testar e Validar (3 minutos)

Após ativar a API e aguardar propagação, execute:

```bash
cd C:\Users\Boanerges\Desktop\Projetos\projeto-stgcn-cpraio
python src/scripts/verify_api_keys.py
```

**Resultado esperado:**
```
📊 CHAVES:
  • Presentes: 3/3
  • Formato válido: 3/3
  • Funcionais: 3/3  ✅

🌐 CONECTIVIDADE:
  • Internet: ✓ OK
  • Arquivo .env: ✓ OK

📝 RECOMENDAÇÕES (0):  ✅
```

---

## 🚨 Checklist de Ações

- [ ] 1. Abrir Google Cloud Console
- [ ] 2. Selecionar projeto 288580115108
- [ ] 3. Ir para APIs & Services > Library
- [ ] 4. Procurar "Generative Language API"
- [ ] 5. Clicar "ENABLE"
- [ ] 6. Aguardar 2-3 minutos
- [ ] 7. (Opcional) Adicionar Billing em Billing > Overview
- [ ] 8. Executar: `python src/scripts/verify_api_keys.py`
- [ ] 9. Verificar resultado (deve mostrar 3/3 funcionais)

---

## 📋 O que cada erro significa

| Erro | Significado | Solução |
|------|-------------|---------|
| **SERVICE_DISABLED** | API não ativada no projeto | Ativar em Google Cloud Console |
| **Quota exceeded (429)** | Limite free tier atingido | Aguardar ou adicionar Billing |
| **Permission Denied** | Chave inválida ou projeto errado | Recrear chave ou mudar projeto |
| **Unauthenticated** | Chave não existe | Gerar nova chave em Google Cloud |

---

## 🔐 Estrutura de Chaves Esperada

Suas chaves em `.env` estão corretas em **formato**, mas precisam de:
1. ✓ Formato: OK (39 caracteres cada)
2. ✓ Projeto: OK (288580115108)
3. ✗ API ativada: **FALTANDO**
4. ✗ Billing: **OPCIONAL MAS RECOMENDADO**

```
.env (current):
GEMINI_KEY_1=AIzaSyDyJ57JME-TAk5-6D15RTpS8oWvqOkmahs
GEMINI_KEY_2=AIzaSyDiyGKvZeWK_6PYgbzOullUYAU_kGc8x6c
GEMINI_KEY_3=AIzaSyA8QcKxXEzY5y9-rWO-Ee4c6dEEC3BCH3o
```

✓ Todas válidas em formato
✓ Todas do mesmo projeto
✗ Falta ativar API no projeto

---

## 🌍 Links Importantes

| Ação | Link |
|------|------|
| **Google Cloud Console** | https://console.cloud.google.com/ |
| **Ativar Generative Language API** | https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com |
| **Configurar Billing** | https://console.cloud.google.com/billing |
| **Documentação Gemini API** | https://ai.google.dev/ |
| **Rate Limits & Quotas** | https://ai.google.dev/gemini-api/docs/rate-limits |
| **Monitorar Uso em Tempo Real** | https://ai.dev/rate-limit |

---

## ⏱️ Cronograma Esperado

```
Agora
  ↓
[5 min] Ativar API no Google Cloud
  ↓
[2-3 min] Aguardar propagação
  ↓
[5 min] (Opcional) Adicionar Billing
  ↓
[1 min] Executar diagnóstico
  ↓
✅ PRONTO! Chaves funcionando
```

**Tempo total: ~15-20 minutos**

---

## 🐛 Se não funcionar após ativar a API

1. **Aguarde mais 5 minutos** - às vezes demora mais para propagar
2. **Limpe cache do navegador** - F5 ou Ctrl+Shift+Delete
3. **Teste em novo terminal** - feche e abra novo PowerShell
4. **Recreie uma chave**:
   - Google Cloud > Credentials > Delete chave
   - Crie nova chave API
   - Copie e cole em `.env`

5. **Se ainda não funcionar, adicione Billing**:
   - Mesmo com FREE TIER, é comum falhar
   - Billing desbloqueará quota muito maior

---

## ✅ Como Confirmar que Está Funcionando

Ao executar o diagnóstico, você deve ver:

```python
✓ GEMINI_KEY_1 VÁLIDA e FUNCIONAL
✓ GEMINI_KEY_2 VÁLIDA e FUNCIONAL  
✓ GEMINI_KEY_3 VÁLIDA e FUNCIONAL

📊 CHAVES:
  • Presentes: 3/3
  • Formato válido: 3/3
  • Funcionais: 3/3  ✅
```

---

## 🎉 Próximas Ações Após Corrigir

Quando as chaves estiverem funcionando:

```bash
# 1. Iniciar a aplicação Streamlit
streamlit run src/app.py

# 2. Gerar previsões
python src/predict.py

# 3. Gerar relatórios tátticos com IA
python src/llm_advisor.py
```

---

## 📞 Resumo Técnico

**Projeto Google Cloud:** `288580115108`

**Problema:** Generative Language API não habilitada

**Status das Chaves:**
- KEY_1: Permissão negada (SERVICE_DISABLED)
- KEY_2: Cota excedida (429 - Free tier saturado)
- KEY_3: Cota excedida (429 - Free tier saturado)

**Solução:** Ativar API + (Opcional) Adicionar Billing

**Tempo estimado:** 15-20 minutos

---

**Última atualização:** 15/01/2026
**Criado por:** Sistema de Diagnóstico API
