# Análise de Requisito e Estimativa
Este projeto utiliza machine learning para analisar requisitos de software e estimar os pontos de história necessários para implementá-los. Os scripts carregam datasets, geram estatísticas, processam o texto dos requisitos, treinam modelos de BERT e de ensemble, e exportam os resultados para um arquivo CSV.

## Configurações Necessárias
Antes de executar o código, algumas constantes e configurações precisam ser ajustadas:

### 1. Diretório dos Datasets
Defina o caminho para o diretório onde os datasets estão localizados:

```python
# Definição de Constantes Globais
DIRETORIO_DATASET_BRUTO = 'D:\Resultados/datasets'
DIRETORIO_DATASET_PROCESSADO = 'D:\Resultados/datasets_processados'
NOME_ARQUIVO_RESULTADOS = 'D:\Resultados/resultados_modelos.csv'
NOME_ARQUIVO_RESULTADOS_ENTRE_DATASETS = 'D:\Resultados/resultados_modelos_ENTRE_DATASETS.csv'
NOME_ARQUIVO_RESULTADOS_MESMO_DATASET = 'D:\Resultados/resultados_modelos_MESMO_DATASET.csv'
NOME_ARQUIVO_BERT_RESULTADOS_ENTRE_DATASETS = 'D:\Resultados/resultados_BERT_modelos_ENTRE_DATASETS.csv'
NOME_ARQUIVO_BERT_RESULTADOS_MESMO_DATASET = 'D:\Resultados/resultados_BERT_modelos_MESMO_DATASET.csv'
```

Os datasets podem ser baixados em: https://github.com/rodrigogrigo/analise-requisitos/tree/main/datasets_all

### 2. Limitação de Registros
Para limitar a quantidade de registros carregados de cada dataset, altere a variável LIMITAR_QUANTIDADE_REGISTROS para True. Quando essa configuração for ativada, ajuste também o valor de QUANTIDADE_REGISTROS_SE_LIMITADO:

```python
LIMITAR_QUANTIDADE_REGISTROS = True
QUANTIDADE_REGISTROS_SE_LIMITADO = 15  # Ajuste conforme necessário
``` 

### 3. Número de Épocas (EPOCHS)
No arquivo bert_evaluation.py, ajuste o valor de EPOCHS conforme necessário para definir o número de épocas para o treinamento do modelo BERT.

```python
EPOCHS = <numero_de_epocas>
```

### Execução
Após configurar as variáveis, execute o script principal com o ambiente configurado:

```bash
myenv\Scripts\python analise_requisito_esforco_storyPoints.py
```

### Resultados
O arquivo com os resultados serão gerados no mesmo local onde estão os fontes, como nome resultados_modelos.csv

### Folds
Os folds com a divisão do dataset para validação cruzada são gerados em /folds. A criação dos folds ocorre apenas 1 vez, caso eles não existam. 
