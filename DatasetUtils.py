from sklearn.base import TransformerMixin
from xgboost import XGBClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from deslib.des import DESKNN
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, VotingRegressor, BaggingRegressor
from nltk.corpus import stopwords
from spellchecker import SpellChecker
import contractions
import spacy
from langdetect import detect as detect_language, LangDetectException
import os
import glob
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm
from unicodedata import normalize
from bs4 import BeautifulSoup
import re
import nltk
import unicodedata
from nltk.stem import PorterStemmer
from collections import Counter
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, root_mean_squared_error
from scipy.stats import pearsonr
import torch
import lightgbm
from transformers import RobertaTokenizer, RobertaModel
from sklearn.base import BaseEstimator, RegressorMixin
from datetime import datetime
from lightgbm import LGBMRegressor
from deslib.des import DESKNN, KNORAU

from desReg.des.DESRegression import DESRegression

# Carregar o modelo de idioma inglês
nlp = spacy.load("en_core_web_sm")

# Função para detectar o idioma usando o Spacy


def detect_language(text):
    doc = nlp(text)
    return doc.lang_


def contar_frases(text):
    # Verifica se o texto não é nulo e contém conteúdo significativo
    if text is not None and len(text.strip()) > 0:
        doc = nlp(text)
        return len(list(doc.sents))
    else:
        return 0  # Retorna 0 se o texto for inválido ou vazio

# Função para contar o número de palavras em uma descrição usando spaCy


def contar_palavras(text):
    # Verifica se o texto não é nulo e contém conteúdo significativo
    if text is not None and len(text.strip()) > 0:
        doc = nlp(text)
        return len(list(doc))
    else:
        return 0  # Retorna 0 se o texto for inválido ou vazio


def detectar_idioma(texto):
    try:
        return detect_language(texto)
    except LangDetectException:
        return 'unknown'


def carregar_dados(fonte_dados):
    # Lendo Dados
    dados_gerais = pd.read_csv(fonte_dados)

    # Aplicar a função de detecção de idioma à coluna 'description' e armazenar o resultado em uma nova coluna
    dados_gerais['lang'] = dados_gerais['description'].apply(
        lambda x: detectar_idioma(str(x)) if pd.notna(x) else 'unknown')

    # Filtrar apenas os registros onde o idioma detectado é inglês
    dados = dados_gerais[dados_gerais['lang'] == 'en']

    # Filtrar registros com storypoint maior que zero e não nulo
    dados = dados[dados['storypoint'].notnull() & (dados['storypoint'] > 0)]

    # Remover registros onde a descrição é NaN ou apenas espaços em branco
    dados = dados[dados['description'].notnull() & (
        dados['description'].str.strip() != '')]

    # Converter descrições restantes para string, garantindo que não há NaN ocultos
    dados['description'] = dados['description'].astype(str)

    # Concatenar 'title' com 'description'
    dados['description'] = dados['title'].astype(
        str) + ' ' + dados['description']

    # Selecionar apenas as colunas 'description', 'storypoint' e 'issuekey'
    dados_filtrados = dados[['description', 'storypoint', 'issuekey']]

    # Remover duplicatas na coluna 'description'
    dados_filtrados = dados_filtrados.drop_duplicates(
        subset=['description'], keep='first')

    return dados_filtrados


def carregar_todos_dados(diretorio_pasta, limitar_registros=False, limite=1000):
    lista_datasets = []
    # Encontrar todos os arquivos .csv no diretório especificado
    arquivos_csv = glob.glob(os.path.join(diretorio_pasta, "*.csv"))

    for arquivo in arquivos_csv:
        dados_filtrados = carregar_dados(arquivo)
        nome_dataset = os.path.basename(arquivo)  # Obtendo o nome do arquivo
        # Adicionando uma nova coluna com o nome do dataset
        dados_filtrados['dataset_name'] = nome_dataset

        # Verificar se deve limitar o número de registros
        if limitar_registros:
            dados_filtrados = dados_filtrados.head(limite)

        lista_datasets.append(dados_filtrados)

    return lista_datasets


def remove_invalid_characters(text):
    # Remove caracteres substitutos ou inválidos
    return ''.join(c for c in text if not unicodedata.category(c).startswith('Cs'))


def preprocessar_descricao(list_descricao):
    descricao_2 = []

    with tqdm(total=len(list_descricao), desc='Processando Descrição') as pbar:
        for descricao in list_descricao:
            if pd.isna(descricao):  # Verifica se o valor é NaN
                descricao_processada = ''  # Substitui NaN por string vazia
            elif not isinstance(descricao, str):  # Verifica se não é string
                descricao_processada = str(descricao)  # Converte para string
            else:
                # Limpa os caracteres inválidos antes de passar para o spaCy
                descricao_limpa = remove_invalid_characters(descricao)

                # Processa o texto com BeautifulSoup
                descricao_processada = BeautifulSoup(
                    descricao_limpa, 'html.parser').get_text()

                # Processa o texto com o spaCy
                try:
                    doc = nlp(descricao_processada)
                    tokens = [t.lemma_.lower() for t in doc if t.pos_ != 'PUNCT'
                              and len(t.lemma_) > 1 and not t.is_stop]
                    descricao_processada = ' '.join(tokens).strip()
                except UnicodeEncodeError as e:
                    # Se houver erro, salva uma string vazia para evitar interrupção
                    descricao_processada = ''

            descricao_2.append(descricao_processada)
            pbar.update(1)

    return descricao_2


def preprocessar_todos_datasets(lista_datasets):
    for dados_filtrados in lista_datasets:
        dados_filtrados['treated_description'] = preprocessar_descricao(
            dados_filtrados['description'].values)

    return lista_datasets


def analisar_datasets(lista_datasets):
    '''
    for i, dados_filtrados in enumerate(lista_datasets):
        # Obtendo o nome do dataset
        dataset_name = dados_filtrados['dataset_name'].iloc[0]
        print(f'\nAnalisando Dataset {i + 1}: {dataset_name}\n')

        descriptions = dados_filtrados['treated_description'].values
        effort_estimations = dados_filtrados['storypoint'].values

        min_effort = min(effort_estimations)
        max_effort = max(effort_estimations)

        effort_estimations = np.array(effort_estimations)

        print(
            f'\nTotal Examples: {len(descriptions)} -- {len(effort_estimations)}\n')
        print(f'\nMin Estimation: {min_effort}')
        print(f'Max Estimation: {max_effort}')

        # Boxplot dos valores de esforço
        plt.figure(figsize=(10, 6))
        plt.boxplot(effort_estimations)
        plt.title(f'{dataset_name}: Boxplot dos Valores de Esforço')
        plt.ylabel('Esforço (em unidades de tempo)')
        plt.show()

        # Calcular os quartis e o IQR
        Q1 = np.percentile(effort_estimations, 25)
        Q3 = np.percentile(effort_estimations, 75)
        IQR = Q3 - Q1

        # Determinar os limites inferior e superior
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identificar outliers
        outliers = effort_estimations[(effort_estimations < lower_bound) | (
            effort_estimations > upper_bound)]

        print(f"Outliers:\n{outliers}")
        print(f"Lower Bound: {lower_bound}")
        print(f"Upper Bound: {upper_bound}")
    '''


def remover_outliers_e_filtrar(lista_datasets):
    '''
    nova_lista_datasets = []

    for i, dados_filtrados in enumerate(lista_datasets):
        # Obtendo o nome do dataset
        dataset_name = dados_filtrados['dataset_name'].iloc[0]
        print(f'\nProcessando Dataset {i + 1}: {dataset_name}\n')

        descriptions = dados_filtrados['treated_description'].values
        effort_estimations = dados_filtrados['storypoint'].values

        # Calcular os quartis e o IQR
        Q1 = np.percentile(effort_estimations, 25)
        Q3 = np.percentile(effort_estimations, 75)
        IQR = Q3 - Q1

        # Determinar os limites inferior e superior
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Remover os outliers
        mask = (effort_estimations >= lower_bound) & (
            effort_estimations <= upper_bound)
        filtered_effort_estimations = effort_estimations[mask]
        filtered_descriptions = descriptions[mask]

        print(
            f'\nTotal Examples After Removing Outliers: {len(filtered_descriptions)} -- {len(filtered_effort_estimations)}\n')
        print(
            f'\nTotal Examples Removed: {len(effort_estimations) - len(filtered_effort_estimations)}\n')

        # Criar um novo DataFrame sem os outliers
        dados_filtrados_sem_outliers = dados_filtrados[mask].copy()

        # Adicionar o DataFrame sem outliers à nova lista
        nova_lista_datasets.append(dados_filtrados_sem_outliers)

        # Boxplot dos valores de esforço sem outliers
        plt.figure(figsize=(10, 6))
        plt.boxplot(filtered_effort_estimations)
        plt.title(
            f'{dataset_name}: Boxplot dos Valores de Esforço sem Outliers')
        plt.ylabel('Esforço (em unidades de tempo)')
        plt.show()

    return nova_lista_datasets
    '''


nltk.download('stopwords')


def exportar_resultados_para_csv(resultados_finais, filename):
    # Verifica se o arquivo já existe
    if os.path.exists(filename):
        # Faz o append dos novos resultados
        resultados_finais.to_csv(filename, mode='a', header=False, index=False)
    else:
        # Cria um novo arquivo com o cabeçalho
        resultados_finais.to_csv(filename, mode='w', header=True, index=False)


def gerar_grafico_valores_reais_vs_estimados(y_test, y_pred, model_name):
    plt.figure(figsize=(10, 6))

    # Plotando os valores reais como pontos
    plt.scatter(range(len(y_test)), y_test,
                label='Valores Reais', color='blue', marker='o')

    # Plotando os valores preditos como pontos
    plt.scatter(range(len(y_pred)), y_pred,
                label='Valores Preditos', color='red', marker='x')

    # Configurações do gráfico
    plt.xlabel('Requisitos')
    plt.ylabel('Valores')
    plt.title(f'Comparação entre Valores Reais e Preditos - {model_name}')
    plt.legend()
    plt.grid(True)
    plt.show()


def imprimir_descricoes_valores(issuekeys, descriptions, y_test, y_pred):
    print(f"{'IssueKey':<15} {'Descrição':<50} {'Valor Real':<20} {'Valor Predito':<20}")
    print("=" * 105)
    for i in range(len(descriptions)):
        descricao = descriptions[i][:50] + \
            ('...' if len(descriptions[i]) > 50 else '')
        print(
            f"{issuekeys[i]:<15} {descricao:<50} {y_test[i]:<20} {y_pred[i]:<20}")


# Carregar stopwords em inglês
stop_words = set(stopwords.words('english'))

# Inicializar corretor ortográfico
spell = SpellChecker()

# Inicializar Stemming
stemmer = PorterStemmer()


def transformar_descricao_em_vetor(descricao, model):
    vetor = np.mean([model.wv[word]
                    for word in descricao if word in model.wv], axis=0)
    return vetor if vetor is not None else np.zeros(model.vector_size)


class DenseTransformer(TransformerMixin):
    """
    Converte arrays esparsos para arrays densos.
    """

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.toarray() if hasattr(X, "toarray") else X


def avaliar_modelosCombinados_em_datasets(lista_datasets, versao_nome):
    # Definição dos modelos base
    models = [
        ('Linear Regression', LinearRegression(), None),
        ('Ridge Regression', Ridge(), {
         'regressor__alpha': [0.1, 1.0, 10.0, 100.0]}),
        ('Lasso Regression', Lasso(max_iter=1000, tol=0.01), {
         'regressor__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]}),
        ('ElasticNet Regression', ElasticNet(max_iter=1000), {
         'regressor__alpha': [0.1, 1.0, 10.0], 'regressor__l1_ratio': [0.1, 0.5, 0.9]}),
        ('SVR', SVR(), {'regressor__C': [
         0.1, 1.0, 10.0], 'regressor__epsilon': [0.01, 0.1, 1.0]}),
        ('KNN', KNeighborsRegressor(), {
         'regressor__n_neighbors': [3, 5, 7, 9]}),
        ('Decision Tree', DecisionTreeRegressor(), {
         'regressor__max_depth': [None, 10, 20, 30]}),
        ('Random Forest', RandomForestRegressor(),
         {'regressor__n_estimators': [100, 200]}),
        ('XGBoost', XGBRegressor(), {'regressor__n_estimators': [
         100, 200], 'regressor__learning_rate': [0.01, 0.1, 0.3]}),
        ('LightGBM', LGBMRegressor(), {'regressor__n_estimators': [
         100, 200], 'regressor__learning_rate': [0.01, 0.1, 0.3]}),
        ('Stacking', StackingRegressor(estimators=[
            ('ridge', Ridge()),
            ('rf', RandomForestRegressor(n_estimators=100)),
            ('xgb', XGBRegressor(n_estimators=100))
        ], final_estimator=Ridge()), None),
        # Adicionando o DESRegression
        # Sem GridSearch para DESRegression
        ('DESRegression', DESRegression(), None)
    ]

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 1), max_features=5_000)),
        # Converte arrays esparsos para densos
        ('to_dense', DenseTransformer()),
        ('normalizer', Normalizer()),  # Normalizar os dados
        ('regressor', None)
    ])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    resultados_completos = []
    predicoes_por_modelo = {}

    for i, dados_filtrados in enumerate(lista_datasets):
        dataset_name = dados_filtrados['dataset_name'].iloc[0]
        print(f'\nAnalisando Dataset {i + 1}: {dataset_name}\n')

        descriptions = dados_filtrados['treated_description'].values
        effort_estimations = dados_filtrados['storypoint'].values
        issuekeys = dados_filtrados['issuekey'].values

        results = {
            'Versao': [],
            'Dataset': [],
            'Model': [],
            'MAE_Mean': [],
            'MAE_Std': [],
            'R2_Mean': [],
            'R2_Std': [],
            'RMSE_Mean': [],
            'RMSE_Std': [],
            'Pearson_Corr_Mean': [],
            'Pearson_Corr_Std': [],
            'Execution_DateTime': []
        }

        for name, model, param_grid in models:
            pipeline.set_params(regressor=model)
            print(f"\nModel: {name}")
            regressor = GridSearchCV(
                pipeline, param_grid, cv=5, scoring='neg_mean_squared_error') if param_grid else pipeline

            list_maes_scores = []
            list_r2_scores = []
            list_rmse_scores = []
            list_pearson_scores = []
            all_y_test = []
            all_y_pred = []
            all_descriptions_test = []
            all_issuekeys_test = []

            for train_index, test_index in kf.split(descriptions):
                X_train, X_test = descriptions[train_index], descriptions[test_index]
                y_train, y_test = effort_estimations[train_index], effort_estimations[test_index]
                issuekeys_test = issuekeys[test_index]

                regressor.fit(X_train, y_train)
                y_pred = regressor.predict(X_test)

                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)

                try:
                    pearson_corr, _ = pearsonr(y_test, y_pred)
                except Exception as e:
                    print(f"Warning: {e}")
                    pearson_corr = float('nan')

                list_maes_scores.append(mae)
                list_r2_scores.append(r2)
                list_rmse_scores.append(rmse)
                list_pearson_scores.append(pearson_corr)

                all_y_test.extend(y_test)
                all_y_pred.extend(y_pred)
                all_descriptions_test.extend(X_test)
                all_issuekeys_test.extend(issuekeys_test)

            predicoes_por_modelo[name] = {
                'issuekeys': all_issuekeys_test,
                'descriptions': all_descriptions_test,
                'y_test': all_y_test,
                'y_pred': all_y_pred
            }

            results['Versao'].append(versao_nome)
            results['Dataset'].append(dataset_name)
            results['Model'].append(name)
            results['MAE_Mean'].append(np.mean(list_maes_scores))
            results['MAE_Std'].append(np.std(list_maes_scores))
            results['R2_Mean'].append(np.mean(list_r2_scores))
            results['R2_Std'].append(np.std(list_r2_scores))
            results['RMSE_Mean'].append(np.mean(list_rmse_scores))
            results['RMSE_Std'].append(np.std(list_rmse_scores))
            results['Pearson_Corr_Mean'].append(np.mean(list_pearson_scores))
            results['Pearson_Corr_Std'].append(np.std(list_pearson_scores))
            results['Execution_DateTime'].append(datetime.now())

        resultados_completos.append(pd.DataFrame(results))

    resultados_finais = pd.concat(resultados_completos, ignore_index=True)

    return resultados_finais, predicoes_por_modelo
