# MVP – Previsão de Chuva (vai chover amanhã?) · Lajeado/RS

Projeto de Machine Learning & Analytics que prevê a **ocorrência de chuva no dia seguinte**
a partir de dados meteorológicos históricos reais. É a continuação aplicada do meu TCC
(plataforma **Ricarol**), onde a previsão de chuva ajusta automaticamente prazos de
contratos sensíveis ao clima.

**Autor:** João Victor de Assis Natividade

---

## Contexto e problema

Saber com antecedência se vai chover permite ajustar decisões operacionais — no caso do
TCC, personalizar o prazo de contratos que dependem do clima. O problema é tratado como
**classificação binária supervisionada** sobre uma **série temporal**:

- **Alvo:** `rain_tomorrow` → 1 se a precipitação do dia seguinte for > 1 mm, senão 0.
- **Entrada:** condições meteorológicas observadas no dia corrente + sinais de sazonalidade.

Prever chuva a um dia, usando apenas dados diários agregados de uma única localidade, é um
problema **genuinamente difícil**. As métricas obtidas são modestas e realistas — fruto de
uma avaliação honesta, sem vazamento de dados.

## Dados

- **Fonte:** [Open-Meteo Historical Weather API](https://open-meteo.com/) (reanálise ERA5).
- **Licença:** CC BY 4.0, dados públicos.
- **Local:** Lajeado/RS (lat −29.4669, lon −51.9644).
- **Período:** 01/01/2019 a 31/05/2026 (2.708 observações diárias, ~7,4 anos).
- **Variáveis:** temperatura média/mín/máx, precipitação, umidade relativa, pressão à
  superfície e velocidade do vento (todas reais — nenhuma feature sintética).

O notebook baixa os dados sozinho via URL pública e roda do início ao fim.

## Metodologia

1. **EDA** — tipos, ausentes, estatísticas, distribuição do alvo, sazonalidade mensal e
   matriz de correlação, cada visualização acompanhada de análise textual.
2. **Preparação** — features de sazonalidade (seno/cosseno do dia do ano) e de memória de
   curto prazo; normalização ajustada **somente no treino**.
3. **Divisão temporal** — últimos 12 meses como teste (dados nunca vistos); ordem
   cronológica preservada.
4. **Baselines** — persistência ("amanhã = hoje") e classe majoritária.
5. **Modelos** — Regressão Logística, Random Forest, Gradient Boosting e Rede Neural Densa
   (TensorFlow).
6. **Otimização** — `GridSearchCV` com `TimeSeriesSplit` (validação cruzada que respeita o
   tempo), critério F1.
7. **Avaliação** — acurácia, precisão, recall, **F1 (principal)**, AUC, matriz de confusão,
   curva ROC, curvas de aprendizado e análise de erros.

### Cuidados contra vazamento de dados
- O alvo é a chuva **futura** (dia seguinte), nunca observada entre as features.
- O `StandardScaler` é ajustado apenas no conjunto de treino.
- A validação cruzada do tuning é temporal (sem look-ahead).

## Resultados

Conjunto de teste: últimos 365 dias (31/05/2025 → 30/05/2026), nunca vistos no treino.
A métrica principal é o **F1** (mais informativo que acurácia sob desbalanceamento de
61,8% / 38,2%).

| Modelo | F1 | Acurácia | AUC |
|---|---|---|---|
| **Rede Neural (MLP)** | **~0,63–0,65** | ~0,68–0,69 | ~0,75 |
| **Random Forest (otimizada)** | **~0,63–0,65** | ~0,68–0,70 | ~0,77 |
| Gradient Boosting | ~0,60–0,65 | ~0,68–0,71 | ~0,75–0,78 |
| Regressão Logística | ~0,60 | ~0,66 | ~0,74 |
| Random Forest | ~0,59 | ~0,72 | ~0,78 |
| Baseline: persistência | 0,578 | 0,660 | — |
| Baseline: classe majoritária | 0,000 | 0,597 | — |

**Empate técnico no topo: Rede Neural (MLP) e Random Forest otimizada**, ambas com F1 em
torno de 0,63–0,65, superando o baseline de persistência (0,578) de forma consistente. O
ganho é modesto, o que reflete a dificuldade real de prever chuva a um dia com dados
diários: a inércia do clima (persistência) já é um baseline forte. As duas têm perfis
complementares — a MLP tende a recall mais alto (captura mais chuvas), a Random Forest
otimizada a precisão mais alta (menos alarmes falsos) —, e a escolha de negócio depende do
custo relativo de cada erro. O fato de duas famílias de modelos distintas convergirem ao
mesmo patamar reforça que esse é o limite do sinal disponível.

> **Nota sobre reprodutibilidade.** As seeds são fixadas no notebook, mas redes neurais
> partem de uma inicialização estocástica; por isso o vencedor por F1 pode alternar entre a
> MLP e a Random Forest otimizada a cada execução, por margens de ~1–2 pontos. Os valores
> acima são apresentados em faixas por esse motivo.

Principais achados da EDA: a chuva tem sazonalidade clara (de ~26% de dias chuvosos em
julho a ~48% em janeiro) e seus preditores isolados mais fortes são a pressão à superfície
e a temperatura mínima — não a umidade, como inicialmente hipotetizado.

## Como executar

1. Abra o [Google Colab](https://colab.research.google.com/).
2. **Arquivo → Fazer upload de notebook** → selecione `MVP_Previsao_Chuva_PUC.ipynb`.
3. **Ambiente de execução → Executar tudo** (Ctrl+F9).

Não é necessário instalar nada nem fazer upload de dados: as bibliotecas já vêm no Colab e
os dados são baixados via URL pública.

### Bibliotecas
`pandas`, `numpy`, `requests`, `matplotlib`, `scikit-learn`, `tensorflow` (todas
pré-instaladas no Colab).

## Estrutura

```
.
├── MVP_Previsao_Chuva_PUC.ipynb   # notebook principal (executável de ponta a ponta)
└── README.md                       # este arquivo
```

## Limitações e próximos passos

- Resolução diária (perde dinâmica intradiária) e uma única localidade.
- Horizonte de 1 dia.
- **Próximos passos:** dados horários; janelas de defasagem mais longas; modelos de
  sequência (LSTM/GRU); usar a previsão operacional do Open-Meteo como feature; calibrar o
  limiar de decisão conforme o custo real de falsos positivos vs. negativos.
