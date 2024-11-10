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
DIRETORIO_DATASETS = 'D:\\Mestrado\\Python\\Projeto\\Datasets\\JIRA-Estimation-Prediction\\storypoint\\IEEE TSE2018\\dataset'
LIMITAR_QUANTIDADE_REGISTROS = False
QUANTIDADE_REGISTROS_SE_LIMITADO = 15
NOME_ARQUIVO_RESULTADOS = 'resultados_modelos.csv'

###########################################################
# Carrega as bases de dados em uma lista
datasetsCarregados = preprocessing.carregar_todos_dados(
    DIRETORIO_DATASETS, LIMITAR_QUANTIDADE_REGISTROS, QUANTIDADE_REGISTROS_SE_LIMITADO)

# Geração de estatísticas do dataset
for dataset in datasetsCarregados:
    preprocessing.gerar_estatisticas_base(dataset)

# Inicia o processamento dos datasets da lista, tratando o texto de cada um deles
datasetsCarregados = preprocessing.preprocessar_todos_datasets(
    datasetsCarregados)
datasetsCarregados[0].head()
###########################################################


###########################################################
# Treinamento e Avaliação dos Modelos BERT

# Executa a Avaliação dos Modelos BERT e Similar
datasets = datasetsCarregados

# Executa o método que realiza o treinamento e teste encima dos datasets, utilizando diversos modelos'
resultados_finais, predicoes_por_modelo = bert_evaluation.avaliar_modelo_bert_em_datasets(
    datasets, VERSAO_NOME_BERT)

# Exportar os resultados para um arquivo CSV
preprocessing.exportar_resultados_para_csv(
    resultados_finais, NOME_ARQUIVO_RESULTADOS)
###########################################################


###########################################################
# Treinamento e Avaliação de Modelos de Ensemble

# Executa a Avaliação dos Modelos BERT e Similar
datasets = datasetsCarregados

# Executa o método que realiza o treinamento e teste encima dos datasets, utilizando diversos modelos'
resultados_finais, predicoes_por_modelo = ensemble_models.avaliar_modelosCombinados_em_datasets(
    datasets, VERSAO_NOME_BERT)

# Exportar os resultados para um arquivo CSV
preprocessing.exportar_resultados_para_csv(
    resultados_finais, NOME_ARQUIVO_RESULTADOS)
###########################################################
