# Análise Fria de Sazonalidade CVLI — Volume Mínimo 10

**Filtro aplicado:** apenas bairros com ≥10 CVLI de histórico (padrões mais robustos)

Bairros incluídos: 85 / 121 (70.2%)
CVLI no dataset filtrado: 1334 de 12339 (10.8%)

## 1. Mês com maior incidência — Análise de picos (volume ≥10)

**Meses que aparecem como picos mais frequentemente (top 5):**

- Mês 3: 10 bairros (11.8%) têm seu pico aqui
- Mês 4: 10 bairros (11.8%) têm seu pico aqui
- Mês 10: 9 bairros (10.6%) têm seu pico aqui
- Mês 5: 9 bairros (10.6%) têm seu pico aqui
- Mês 8: 7 bairros (8.2%) têm seu pico aqui

**Conclusão:** Mês 3 é o de maior incidência "padrão" (10 / 85 bairros = 11.8%)

## 2. Horário com maior incidência — Análise de picos (volume ≥10)

**Horários que aparecem como picos mais frequentemente (top 5):**

- Hora 20: 12 bairros (14.1%) têm seu pico aqui
- Hora 19: 10 bairros (11.8%) têm seu pico aqui
- Hora 16: 9 bairros (10.6%) têm seu pico aqui
- Hora 21: 7 bairros (8.2%) têm seu pico aqui
- Hora 13: 6 bairros (7.1%) têm seu pico aqui

**Conclusão:** Hora 20 (horário de pico padrão) aparece em 12 / 85 bairros = 14.1%

## 3. Dia da semana com maior incidência — Análise de picos (volume ≥10)

**Dias que aparecem como picos mais frequentemente (top 5):**

- Domingo: 27 bairros (31.8%) têm seu pico neste dia
- Sexta-feira: 14 bairros (16.5%) têm seu pico neste dia
- Quarta-feira: 14 bairros (16.5%) têm seu pico neste dia
- Sábado: 13 bairros (15.3%) têm seu pico neste dia
- Segunda-feira: 7 bairros (8.2%) têm seu pico neste dia

**Conclusão:** Domingo é o padrão de pico mais comum (27 / 85 bairros = 31.8%)

## 4. Bairros com padrão sazonal consistente ("estações do ano") — volume ≥10

**Bairros com maior consistência sazonal (padrão repetido ano a ano) — volume ≥10:**

1. AUTRAN NUNES / Fortaleza: consistência=1.00, total_cvli=7
2. ITAPERI / Fortaleza: consistência=1.00, total_cvli=8
3. ENGENHEIRO LUCIANO CAVALCANTE / Fortaleza: consistência=1.00, total_cvli=7
4. CONJUNTO CEARÁ I / Fortaleza: consistência=0.86, total_cvli=8
5. MANUEL DIAS BRANCO / Fortaleza: consistência=0.86, total_cvli=8
6. PARANGABA / Fortaleza: consistência=0.86, total_cvli=8
7. VILA PERI / Fortaleza: consistência=0.85, total_cvli=13
8. MESSEJANA / Fortaleza: consistência=0.84, total_cvli=31
9. CONJUNTO CEARÁ II / Fortaleza: consistência=0.84, total_cvli=7
10. MOURA BRASIL / Fortaleza: consistência=0.84, total_cvli=6
11. GRANJA PORTUGAL / Fortaleza: consistência=0.83, total_cvli=20
12. JANGURUSSU / Fortaleza: consistência=0.82, total_cvli=40
13. FARIAS BRITO / Fortaleza: consistência=0.82, total_cvli=7
14. BARROSO / Fortaleza: consistência=0.81, total_cvli=21
15. JOSÉ BONIFÁCIO / Fortaleza: consistência=0.80, total_cvli=6
16. GRANJA LISBOA / Fortaleza: consistência=0.80, total_cvli=32
17. ALTO DA BALANÇA / Fortaleza: consistência=0.80, total_cvli=8
18. BARRA DO CEARÁ / Fortaleza: consistência=0.80, total_cvli=39
19. EDSON QUEIROZ / Fortaleza: consistência=0.79, total_cvli=16
20. SIQUEIRA / Fortaleza: consistência=0.78, total_cvli=22

(Consistência próxima a 1.0 = padrão muito forte; próxima a 0 = padrão fraco)

---
Sazonalidade volume filtrado salvo em: outputs\docs\cvli_bairros_volume_analysis_min10.csv