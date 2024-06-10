# %pip install langdetect
# %pip install pandas
# %pip install spacy
# %pip install scikit-learn
# %pip install bs4
# %pip install nltk

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold, GridSearchCV
from collections import Counter
from nltk.stem import PorterStemmer
import nltk
import re
from bs4 import BeautifulSoup
from unicodedata import normalize
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from langdetect import detect
import pandas as pd
import spacy

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


# Lendo Dados
dados_gerais = pd.read_csv(
    'C:\\Users\\rodri\\Downloads\\JiraRepos.JiraEcosystem.csv')

# Aplicar a função de detecção de idioma à coluna 'fields.description' e armazenar o resultado em uma nova coluna
dados_gerais['lang'] = dados_gerais['fields.description'].apply(
    lambda x: detect_language(str(x)))

# Filtrar apenas os registros onde o idioma detectado é inglês
dados = dados_gerais[dados_gerais['lang'] == 'en']

# Selecionar apenas as colunas 'fields.description', 'fields.timeestimate' e 'id'
dados_filtrados = dados[['fields.description', 'fields.timeestimate', 'id']]

dados_filtrados.head()


def preprocessar_descricao(list_descricao, remove_punctuation=True, lower_case=True, remove_rare_words=False, remove_common_words=False, stem=False, custom_stop_words=None):
    descricao_2 = []
    stop_words = set(nlp.Defaults.stop_words)
    if custom_stop_words:
        stop_words.update(custom_stop_words)

    # Construir o vocabulário
    all_words = []
    for descricao in list_descricao:
        descricao_limpa = BeautifulSoup(descricao, 'html.parser').get_text()
        if lower_case:
            descricao_limpa = descricao_limpa.lower()
        if remove_punctuation:
            descricao_limpa = re.sub(r'[^\w\s]', '', descricao_limpa)
        words = descricao_limpa.split()
        all_words.extend(words)

    word_counts = Counter(all_words)

    # Remover palavras raras e comuns, se necessário
    if remove_rare_words:
        all_words = [
            word for word in all_words if word_counts[word] > remove_rare_words]
    if remove_common_words:
        common_words = [word for word,
                        count in word_counts.most_common(remove_common_words)]
        all_words = [word for word in all_words if word not in common_words]

    # Stemming ou lematização
    if stem:
        ps = PorterStemmer()
        all_words = [ps.stem(word) for word in all_words]
    else:
        doc = nlp(' '.join(all_words))
        all_words = [token.lemma_ for token in doc]

    # Remover stop words
    all_words = [word for word in all_words if word not in stop_words]

    # Construir o corpus processado
    with tqdm(total=len(list_descricao), desc='Processando Descrições') as pbar:
        for descricao in list_descricao:
            descricao_limpa = BeautifulSoup(
                descricao, 'html.parser').get_text()
            if lower_case:
                descricao_limpa = descricao_limpa.lower()
            if remove_punctuation:
                descricao_limpa = re.sub(r'[^\w\s]', '', descricao_limpa)
            words = descricao_limpa.split()
            if remove_rare_words:
                words = [word for word in words if word_counts[word]
                         > remove_rare_words]
            if remove_common_words:
                words = [word for word in words if word not in common_words]
            if stem:
                words = [ps.stem(word) for word in words]
            else:
                doc = nlp(' '.join(words))
                words = [token.lemma_ for token in doc]
            words = [word for word in words if word not in stop_words]
            descricao_processada = ' '.join(words).strip()
            descricao_2.append(descricao_processada)
            pbar.update(1)

    return descricao_2


# Definindo os dados de entrada e saída
X = preprocessar_descricao(dados_filtrados['fields.description'].values)
y = dados_filtrados['fields.timeestimate'].values


# Definindo os modelos que deseja avaliar
models = [
    ('Linear Regression', LinearRegression(), {}),
    ('Ridge Regression', Ridge(), {
     'regressor__alpha': [0.1, 1.0, 10.0, 100.0]}),
    ('Lasso Regression', Lasso(max_iter=10000), {
     'regressor__alpha': [0.1, 1.0, 10.0, 100.0]}),
    ('ElasticNet Regression', ElasticNet(max_iter=10000), {'regressor__alpha': [
     0.1, 1.0, 10.0, 100.0], 'regressor__l1_ratio': [0.1, 0.5, 0.9]}),
    ('SVR', SVR(), {'regressor__C': [
     0.1, 1.0, 10.0], 'regressor__epsilon': [0.01, 0.1, 1.0]}),
    ('KNN', KNeighborsRegressor(), {'regressor__n_neighbors': [3, 5, 7, 9]}),
    ('Decision Tree', DecisionTreeRegressor(), {
     'regressor__max_depth': [None, 10, 20, 30]})
]

# Criando um pipeline com vetorização de texto e modelo de regressão
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),  # Convertendo texto em TF-IDF
    ('regressor', None)  # O modelo será adicionado dinamicamente
])

# Usando StratifiedKFold para validação cruzada
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Avaliando cada modelo usando validação cruzada
for name, model, param_grid in models:
    pipeline.set_params(regressor=model)
    if param_grid:
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=kf, scoring='neg_mean_squared_error')
        grid_search.fit(X, y)
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        rmse_score = (-best_score)**0.5
        print(f"Model: {name}")
        print(f"Best Parameters: {best_params}")
        print(f"Best RMSE: {rmse_score}")
    else:
        scores = cross_val_score(
            pipeline, X, y, cv=kf, scoring='neg_mean_squared_error')
        rmse_scores = (-scores)**0.5
        print(f"Model: {name}")
        print(
            f"Mean RMSE: {rmse_scores.mean()}, Std RMSE: {rmse_scores.std()}")
    print("-" * 30)
