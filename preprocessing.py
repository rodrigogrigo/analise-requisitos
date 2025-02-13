import spacy
from langdetect import detect, LangDetectException
import os
import glob
from tqdm import tqdm
from bs4 import BeautifulSoup
import unicodedata
import re
import pandas as pd
from collections import Counter
import numpy as np

# Carregar o modelo de idioma inglês
nlp = spacy.load("en_core_web_sm")

# Sequência de Fibonacci para verificação dos story points
fibonacci_sequence = [1, 2, 3, 5, 8, 13, 21]

def detectar_idioma(texto):
    try:
        return detect(texto)
    except LangDetectException:
        return 'unknown'
    
def uniformizar_storypoints(lista_datasets):
    """
    Dado uma lista de DataFrames (cada um representando um dataset já com outliers removidos),
    filtra cada dataset para que nenhum registro possua um storypoint maior que o menor valor máximo entre todos os datasets.
    
    Retorna:
      - nova_lista_datasets: lista dos DataFrames filtrados
      - min_max_sp: o valor máximo uniforme definido (mínimo dos máximos originais)
    """
    # Obter o máximo de storypoint de cada dataset
    max_sp_list = [df['storypoint'].max() for df in lista_datasets if not df.empty]
    
    if not max_sp_list:
        return lista_datasets, None  # Caso não haja dados
    
    # Encontrar o menor valor máximo entre todos os datasets
    min_max_sp = min(max_sp_list)
    
    # Filtrar cada dataset removendo os registros com storypoint maiores que min_max_sp
    nova_lista_datasets = []
    for df in lista_datasets:
        df_uniform = df[df['storypoint'] <= min_max_sp].copy()
        nova_lista_datasets.append(df_uniform)
    
    return nova_lista_datasets, min_max_sp


def contar_frases(text):
    if text is not None and len(text.strip()) > 0:
        doc = nlp(text)
        return len(list(doc.sents))
    else:
        return 0

def contar_palavras_spacy(text):
    doc = nlp(text)
    return len([token for token in doc if not token.is_punct and not token.is_space])

def carregar_dados(fonte_dados):
    dados_gerais = pd.read_csv(fonte_dados)
    
    # Detectar idioma
    dados_gerais['lang'] = dados_gerais['description'].apply(
        lambda x: detectar_idioma(str(x)) if pd.notna(x) else 'unknown')
    
    # Filtrar apenas descrições em inglês
    dados = dados_gerais[dados_gerais['lang'] == 'en']
    
    # Filtrar registros válidos
    dados = dados[dados['storypoint'].notnull() & (dados['storypoint'] > 0)]
    dados = dados[dados['description'].notnull() & (dados['description'].str.strip() != '')]
    
    # Concatenar título e descrição
    dados['description'] = dados['description'].astype(str)
    dados['description'] = dados['title'].astype(str) + ' ' + dados['description']
    
    # Calcular IQR para remoção de outliers
    Q1 = dados['storypoint'].quantile(0.25)
    Q3 = dados['storypoint'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Remover outliers
    dados = dados[(dados['storypoint'] >= lower_bound) & (dados['storypoint'] <= upper_bound)]
    
    # Selecionar colunas relevantes e remover duplicatas
    dados_filtrados = dados[['description', 'storypoint', 'issuekey']]
    dados_filtrados = dados_filtrados.drop_duplicates(subset=['description'], keep='first')
    
    return dados_filtrados

def carregar_todos_dados(diretorio_bruto, diretorio_processado, limitar_registros=False, limite=1000):
    os.makedirs(diretorio_processado, exist_ok=True)
    lista_datasets_comuns = []
    lista_datasets_bert = []

    arquivos_brutos = glob.glob(os.path.join(diretorio_bruto, "*.csv"))
    for arquivo in arquivos_brutos:
        nome_dataset = os.path.splitext(os.path.basename(arquivo))[0]
        caminho_comum = os.path.join(diretorio_processado, f"{nome_dataset}_tradicional_processed.csv")
        caminho_bert = os.path.join(diretorio_processado, f"{nome_dataset}_bert_processed.csv")

        if os.path.exists(caminho_comum):
            print(f"Carregando dataset processado (comum): {nome_dataset}")
            dados_comuns = pd.read_csv(caminho_comum)
            lista_datasets_comuns.append(dados_comuns)
        else:
            print(f"Processando dataset bruto (comum): {nome_dataset}")
            dados_filtrados = carregar_dados(arquivo)  # Aqui já ocorre a remoção de outliers via IQR
            dados_filtrados['dataset_name'] = nome_dataset
            if limitar_registros:
                dados_filtrados = dados_filtrados.head(limite)
            lista_para_preprocessar = [dados_filtrados]
            dados_processados_comum = preprocessar_todos_datasets(lista_para_preprocessar, baseBert=False)[0]
            dados_processados_comum.to_csv(caminho_comum, index=False)
            print(f"Dataset processado salvo (comum): {caminho_comum}")
            lista_datasets_comuns.append(dados_processados_comum)

        if os.path.exists(caminho_bert):
            print(f"Carregando dataset processado (BERT): {nome_dataset}")
            dados_bert = pd.read_csv(caminho_bert)
            lista_datasets_bert.append(dados_bert)
        else:
            print(f"Processando dataset bruto (BERT): {nome_dataset}")
            dados_filtrados = carregar_dados(arquivo)
            dados_filtrados['dataset_name'] = nome_dataset
            if limitar_registros:
                dados_filtrados = dados_filtrados.head(limite)
            lista_para_preprocessar = [dados_filtrados]
            dados_processados_bert = preprocessar_todos_datasets(lista_para_preprocessar, baseBert=True)[0]
            dados_processados_bert.to_csv(caminho_bert, index=False)
            print(f"Dataset processado salvo (BERT): {caminho_bert}")
            lista_datasets_bert.append(dados_processados_bert)

    # Uniformizar os storypoints nos datasets "comuns"
    lista_datasets_comuns, min_max_sp = uniformizar_storypoints(lista_datasets_comuns)
    print(f"Valor máximo uniforme definido para storypoints: {min_max_sp}")
    
    # Se desejar, pode aplicar o mesmo para os datasets BERT:
    lista_datasets_bert, _ = uniformizar_storypoints(lista_datasets_bert)

    return lista_datasets_comuns, lista_datasets_bert


def remove_invalid_characters(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def preprocessar_descricao(list_descricao, baseBert=False):
    descricao_2 = []
    with tqdm(total=len(list_descricao), desc='Processando Descrição') as pbar:
        for descricao in list_descricao:
            if pd.isna(descricao):
                descricao_processada = ''
            elif not isinstance(descricao, str):
                descricao_processada = str(descricao)
            else:
                descricao_limpa = remove_invalid_characters(descricao)
                descricao_processada = BeautifulSoup(descricao_limpa, 'html.parser').get_text()
                if not baseBert:
                    try:
                        doc = nlp(descricao_processada)
                        tokens = [t.lemma_.lower() for t in doc if t.pos_ != 'PUNCT' and len(t.lemma_) > 1 and not t.is_stop]
                        descricao_processada = ' '.join(tokens).strip()
                    except UnicodeEncodeError as e:
                        descricao_processada = ''
            descricao_2.append(descricao_processada)
            pbar.update(1)
    return descricao_2

def preprocessar_todos_datasets(lista_datasets, baseBert=False):
    for dados_filtrados in lista_datasets:
        dados_filtrados['treated_description'] = preprocessar_descricao(dados_filtrados['description'].values, baseBert)
    return lista_datasets

def exportar_resultados_para_csv(resultados_finais, filename):
    if os.path.exists(filename):
        resultados_finais.to_csv(filename, mode='a', header=False, index=False)
    else:
        resultados_finais.to_csv(filename, mode='w', header=True, index=False)

def gerar_estatisticas_base(lista_datasets):
    estatisticas_lista = []
    for dados_filtrados in lista_datasets:
        nome_base = dados_filtrados['dataset_name'].iloc[0]
        total_registros = len(dados_filtrados)
        contagem_palavras = dados_filtrados['description'].apply(contar_palavras_spacy)
        media_palavras = np.mean(contagem_palavras)
        desvio_palavras = np.std(contagem_palavras)
        story_points = dados_filtrados['storypoint'].values
        distribuicao_story_points = Counter(story_points)
        distribuicao_fibonacci = {ponto: distribuicao_story_points[ponto] for ponto in fibonacci_sequence if ponto in distribuicao_story_points}
        story_point_min = dados_filtrados['storypoint'].min()
        story_point_max = dados_filtrados['storypoint'].max()
        story_point_mean = dados_filtrados['storypoint'].mean()
        story_point_median = dados_filtrados['storypoint'].median()
        estatisticas_lista.append({
            'Base de Dados': nome_base,
            'Total de Registros': total_registros,
            'Média de Palavras por Registro': media_palavras,
            'Desvio Padrão de Palavras por Registro': desvio_palavras,
            'Story Point Min': story_point_min,
            'Story Point Max': story_point_max,
            'Story Point Mean': story_point_mean,
            'Story Point Median': story_point_median
        })
    estatisticas_df = pd.DataFrame(estatisticas_lista)
    return estatisticas_df