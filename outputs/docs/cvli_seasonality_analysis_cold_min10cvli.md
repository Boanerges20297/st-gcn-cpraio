# Análise Fria de Sazonalidade CVLI — Volume Mínimo 10

**Filtro aplicado:** apenas bairros com ≥10 CVLI de histórico (padrões mais robustos)

Bairros incluídos: 481 / 5008 (9.6%)
CVLI no dataset filtrado: 7795 de 22839 (34.1%)

## 1. Mês com maior incidência — Análise de picos (volume ≥10)

**Meses que aparecem como picos mais frequentemente (top 5):**

- Mês 1: 68 bairros (14.1%) têm seu pico aqui
- Mês 3: 51 bairros (10.6%) têm seu pico aqui
- Mês 4: 48 bairros (10.0%) têm seu pico aqui
- Mês 7: 43 bairros (8.9%) têm seu pico aqui
- Mês 2: 42 bairros (8.7%) têm seu pico aqui

**Conclusão:** Mês 1 é o de maior incidência "padrão" (68 / 481 bairros = 14.1%)

## 2. Horário com maior incidência — Análise de picos (volume ≥10)

**Horários que aparecem como picos mais frequentemente (top 5):**

- Hora 0: 59 bairros (12.3%) têm seu pico aqui
- Hora 22: 39 bairros (8.1%) têm seu pico aqui
- Hora 15: 37 bairros (7.7%) têm seu pico aqui
- Hora 21: 37 bairros (7.7%) têm seu pico aqui
- Hora 1: 36 bairros (7.5%) têm seu pico aqui

**Conclusão:** Hora 0 (horário de pico padrão) aparece em 59 / 481 bairros = 12.3%

## 3. Dia da semana com maior incidência — Análise de picos (volume ≥10)

**Dias que aparecem como picos mais frequentemente (top 5):**

- quarta-feira: 86 bairros (17.9%) têm seu pico neste dia
- quinta-feira: 77 bairros (16.0%) têm seu pico neste dia
- domingo: 76 bairros (15.8%) têm seu pico neste dia
- sexta-feira: 74 bairros (15.4%) têm seu pico neste dia
- segunda-feira: 64 bairros (13.3%) têm seu pico neste dia

**Conclusão:** quarta-feira é o padrão de pico mais comum (86 / 481 bairros = 17.9%)

## 4. Bairros com padrão sazonal consistente ("estações do ano") — volume ≥10

**Bairros com maior consistência sazonal (padrão repetido ano a ano) — volume ≥10:**

1. Zona Rural / São Gonçalo do Amarante: consistência=1.00, total_cvli=9
2. Jose Walter / Fortaleza: consistência=1.00, total_cvli=9
3. Pedras / Fortaleza: consistência=1.00, total_cvli=9
4. Barroso I / Fortaleza: consistência=1.00, total_cvli=9
5. Ellery / Fortaleza: consistência=1.00, total_cvli=9
6. Alto Alegre 1 / Maracanaú: consistência=1.00, total_cvli=7
7. Aeroporto / Juazeiro do Norte: consistência=1.00, total_cvli=8
8. Zona Rural / Pacajus: consistência=1.00, total_cvli=10
9. São Vicente / Crateús: consistência=1.00, total_cvli=10
10. Centro / Maracanaú: consistência=1.00, total_cvli=8
11. Cidade Nova / Crateús: consistência=0.90, total_cvli=9
12. Rodoviária / Tianguá: consistência=0.89, total_cvli=7
13. Mucunã / Baturité: consistência=0.89, total_cvli=7
14. Cocobo / Iguatu: consistência=0.89, total_cvli=7
15. Zona Rural / Baturité: consistência=0.87, total_cvli=10
16. Alto São João / Pacatuba: consistência=0.87, total_cvli=10
17. Parque das Nações / Caucaia: consistência=0.86, total_cvli=9
18. Jardim Primavera / Cascavel: consistência=0.86, total_cvli=8
19. Jereissati II / Maracanaú: consistência=0.86, total_cvli=8
20. Zona Rural / Itapipoca: consistência=0.86, total_cvli=8

(Consistência próxima a 1.0 = padrão muito forte; próxima a 0 = padrão fraco)

---
Sazonalidade volume filtrado salvo em: outputs\docs\cvli_bairros_volume_analysis_min10.csv