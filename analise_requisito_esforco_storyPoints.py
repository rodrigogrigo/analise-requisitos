import ensemble_models
import bert_evaluation
import preprocessing
import subprocess
import sys

import subprocess
import sys
import importlib


def install(package):
    try:
        # Tenta importar o pacote para verificar se já está instalado
        importlib.import_module(package)
        print(f"'{package}' já está instalado.")
    except ImportError:
        # Se a importação falhar, instala o pacote
        print(f"Instalando '{package}'...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package])


# Instalação das bibliotecas
install("langdetect")
install("pandas")
install("spacy")
install("scikit-learn")
install("bs4")
install("nltk")
install("xgboost")
install("catboost")
install("transformers")
install("torch")
install("lightgbm")
install("contractions")
install("pyspellchecker")
install("gensim")
install("deslib")
install("desReg")
install("numpy")
install("scipy")
install("transformers[torch]")

# Importações das Bibliotecas

importlib.reload(preprocessing)
importlib.reload(bert_evaluation)
importlib.reload(ensemble_models)

# Definição de Constantes Globais
VERSAO_NOME_BERT = "V_BERT"
VERSAO_NOME_ENSEMBLE = "V_ENSEMBLE"
DIRETORIO_DATASET_BRUTO = 'D:\\Mestrado\\Python\\Projeto\\Datasets\\JIRA-Estimation-Prediction\\storypoint\\IEEE TSE2018\\dataset'
DIRETORIO_DATASET_PROCESSADO = 'D:\\Mestrado\\Python\\Projeto\\Datasets\\JIRA-Estimation-Prediction\\storypoint\\IEEE TSE2018\\dataset_processado'
LIMITAR_QUANTIDADE_REGISTROS = False
QUANTIDADE_REGISTROS_SE_LIMITADO = 15
NOME_ARQUIVO_RESULTADOS = 'resultados_modelos.csv'

###########################################################
# Carrega as bases de dados em uma lista
datasetsComuns, datasetsBert = preprocessing.carregar_todos_dados(
    DIRETORIO_DATASET_BRUTO, DIRETORIO_DATASET_PROCESSADO, LIMITAR_QUANTIDADE_REGISTROS, QUANTIDADE_REGISTROS_SE_LIMITADO)

# Geração de estatísticas do dataset
for dataset in datasetsComuns:
    preprocessing.gerar_estatisticas_base(dataset)

for dataset in datasetsBert:
    preprocessing.gerar_estatisticas_base(dataset)
###########################################################


###########################################################
# Treinamento e Avaliação dos Modelos BERT

# Executa o método que realiza o treinamento e teste encima dos datasets, utilizando diversos modelos'
resultados_finais, predicoes_por_modelo = bert_evaluation.avaliar_modelo_bert_em_datasets(
    datasetsBert, VERSAO_NOME_BERT)

# Exportar os resultados para um arquivo CSV
preprocessing.exportar_resultados_para_csv(
    resultados_finais, NOME_ARQUIVO_RESULTADOS)
###########################################################


###########################################################
# Treinamento e Avaliação de Modelos de Ensemble

# Executa o método que realiza o treinamento e teste encima dos datasets, utilizando diversos modelos'
resultados_finais, predicoes_por_modelo = ensemble_models.avaliar_modelosCombinados_em_datasets(
    datasetsComuns, VERSAO_NOME_ENSEMBLE)

# Exportar os resultados para um arquivo CSV
preprocessing.exportar_resultados_para_csv(
    resultados_finais, NOME_ARQUIVO_RESULTADOS)
###########################################################
