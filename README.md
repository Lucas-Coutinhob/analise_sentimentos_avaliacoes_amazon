# Análise de Sentimento - Avaliações Amazon & Mercado Livre (pt-BR)

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.3-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3.9-green)
![pysentimiento](https://img.shields.io/badge/pysentimiento-0.7-red)
![Status](https://img.shields.io/badge/Status-Concluído-brightgreen)

---

## Contexto de Negócio

Em marketplaces como Amazon e Mercado Livre, avaliações de clientes são a principal fonte de feedback não estruturado sobre produtos. Compreender o sentimento dessas avaliações permite que equipes de produto identifiquem pontos de dor, marketing segmente comunicação por perfil de satisfação, e vendedores monitorem reputação de forma automatizada.

Este projeto classifica automaticamente o sentimento de **+67 mil avaliações reais** em português, escritas por consumidores brasileiros em 43 categorias de produtos, comparando o desempenho de múltiplos algoritmos de classificação.

---

## Principais Resultados

| Métrica | Valor |
|---------|-------|
| Reviews analisados | 67.922 |
| Categorias de produto | 43 (agrupadas em 8 macro-categorias) |
| Plataformas | Amazon Brasil + Mercado Livre |
| Modelo final | Logistic Regression + Bag of Words |
| Macro F1 (otimizado) | **0.76** |
| Accuracy | **0.86** |
| F1 Positivo / Neutro / Negativo | 0.93 / 0.59 / 0.76 |

### Insight Principal

> **Clientes insatisfeitos escrevem 60% mais que satisfeitos** (113 vs 71 caracteres em média). Reviews longos (101-300 chars) concentram 24.8% de negatividade, quase 8x mais que reviews curtos (≤10 chars: 3%). A extensão do review é, por si só, um sinal de sentimento.

---

## Descobertas de Negócio

- **Produtos mais criticados:** drones (30.5% negativo), smartwatches (29%) e ar-condicionado portátil (26.8%). Padrão comum: promessa exagerada vs entrega real.
- **Produtos mais amados:** fritadeiras (85% positivo), monitores (84.3%) e vestidos (83.3%). Produtos com expectativa clara e entrega previsível.
- **Amazon vs Mercado Livre:** Amazon tem menos negatividade (9.3% vs 14.6%) mas mais neutralidade (22.8% vs 14.6%). Consumidores da Amazon são mais descritivos; do Mercado Livre, mais polarizados.
- **Bigramas com negação** ("não gostei", "não recomendo") são os indicadores mais fortes de sentimento negativo, validando a inclusão de n-grams na vetorização.

---

## Estrutura do Projeto
```
analise_sentimentos_avaliacoes_Amazon/
├── dados/
│   ├── DOWNLOAD_AQUI.txt
│   ├── am_scrape_final.csv
│   └── ml_scrape_final.csv
├── notebooks/
│   └── analise_sentimentos.ipynb
├── modelo/
│   ├── modelo_sentimento.pkl
│   ├── vetorizador_bow.pkl
│   └── pipeline_completo.pkl
├── outputs/
│   ├── distribuicao_sentimentos.png
│   ├── sentimento_por_plataforma.png
│   ├── sentimento_por_categoria.png
│   ├── top_bottom_pesquisas.png
│   ├── wordclouds_sentimento.png
│   ├── tamanho_review_sentimento.png
│   ├── sentimento_por_faixa_tamanho.png
│   ├── matriz_confusao_melhor_modelo.png
│   ├── comparacao_antes_depois_tuning.png
│   └── palavras_mais_relevantes_por_classe.png
├── .gitignore
├── README.md
└── requirements.txt
```
---

## Pipeline Técnico

1. Carga e unificação de dois datasets (Amazon + Mercado Livre) com padronização de colunas
2. Limpeza: remoção de 1.716 registros (duplicatas de scraping + reviews ≤2 chars)
3. Engenharia de atributos: macro-categorias de varejo, métricas textuais, coluna de plataforma
4. Rotulação automática de sentimento com modelo BERT pré-treinado (pysentimiento pt-BR)
5. Validação da rotulação: 24 testes manuais + análise de confiança (98.3% com confiança >0.5)
6. EDA completa: sentimento por plataforma, categoria, pesquisa e tamanho do review
7. Pré-processamento NLP: limpeza textual, remoção de stopwords (250), stemming RSLP
8. Vetorização: CountVectorizer e TF-IDF com unigramas + bigramas (vocabulário de 10.000 termos)
9. Modelagem: 8 combinações (4 modelos x 2 vetorizações) avaliadas com Macro F1
10. Otimização de hiperparâmetros via GridSearchCV (5-fold CV estratificado)
11. Análise de erros e interpretabilidade via coeficientes da Logistic Regression
12. Serialização do pipeline completo para inferência em novos dados

---

## Stack Tecnológica

| Ferramenta | Uso |
|------------|-----|
| Python 3.12 | Linguagem principal |
| pandas / NumPy | Manipulação e análise de dados |
| scikit-learn | Modelagem, vetorização, avaliação, otimização |
| NLTK | Stopwords pt-BR, stemming RSLP |
| pysentimiento | Rotulação de sentimento com BERT pt-BR |
| matplotlib / seaborn | Visualizações |
| WordCloud | Nuvens de palavras por sentimento |
| joblib | Serialização do pipeline |

---

## Dataset

- **Nome:** Avaliações em Português - Amazon e Mercado Livre
- **Fonte:** [Kaggle - Vitor Sampaio](https://www.kaggle.com/datasets/sampaiovitor/avaliaes-em-portugus-amazon-e-mercado-livre)
- **Volume:** 69.749 reviews brutos -> 67.922 após limpeza
- **Período:** Coletado via web scraping
- **Cobertura:** 43 categorias de produto, 1.804 produtos únicos

---

## Como Reproduzir

```bash
# 1. Clonar o repositório
git clone https://github.com/Lucas-Coutinhob/analise-sentimentos-avaliacoes-Amazon.git
cd analise-sentimentos-avaliacoes-Amazon

# 2. Criar ambiente virtual
conda create -n sentimentos python=3.12 -y
conda activate sentimentos

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Baixar recursos NLP
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"
python -m spacy download pt_core_news_sm

# 5. Baixar o dataset
# Acesse: https://www.kaggle.com/datasets/sampaiovitor/avaliaes-em-portugus-amazon-e-mercado-livre
# Salve os CSVs na pasta dados/

# 6. Executar o notebook
jupyter notebook notebooks/analise_sentimentos.ipynb
```

---

## Autor

**Lucas Coutinho Boros**
Cientista de Dados em Formação | Bacharelado em Ciência de Dados e IA - IESB

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Lucas%20Boros-blue?logo=linkedin)](https://www.linkedin.com/in/lucas-coutinho-boros)
[![GitHub](https://img.shields.io/badge/GitHub-Lucas--Coutinhob-black?logo=github)](https://github.com/Lucas-Coutinhob)
📧 lucas.boros@live.com
