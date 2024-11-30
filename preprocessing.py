import spacy
from langdetect import detect as LangDetectException
import os
import glob
from tqdm import tqdm
from bs4 import BeautifulSoup
import unicodedata
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import numpy as np

# Carregar o modelo de idioma inglês
nlp = spacy.load("en_core_web_sm")

# Sequência de Fibonacci para verificação dos story points
fibonacci_sequence = [1, 2, 3, 5, 8, 13, 21]


def detectar_idioma(texto):
    try:
        doc = nlp(texto)
        return doc.lang_
    except LangDetectException:
        return 'unknown'

# Verifica se o texto não é nulo e contém conteúdo significativo


def contar_frases(text):
    if text is not None and len(text.strip()) > 0:
        doc = nlp(text)
        return len(list(doc.sents))
    else:
        return 0  # Retorna 0 se o texto for inválido ou vazio

# Função para contar o número de palavras em uma descrição usando spaCy


def contar_palavras_spacy(text):
    doc = nlp(text)
    return len([token for token in doc if not token.is_punct and not token.is_space])


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


def carregar_todos_dados(diretorio_bruto, diretorio_processado, limitar_registros=False, limite=1000):
    """
    Carrega datasets processados (BERT e comum), ou processa os brutos e salva os processados.
    """
    import os
    import glob

    # Garantir que o diretório de datasets processados exista
    os.makedirs(diretorio_processado, exist_ok=True)

    lista_datasets_comuns = []
    lista_datasets_bert = []

    # Encontrar todos os arquivos brutos no diretório bruto
    arquivos_brutos = glob.glob(os.path.join(diretorio_bruto, "*.csv"))

    for arquivo in arquivos_brutos:
        # Nome base do dataset (sem extensão)
        nome_dataset = os.path.splitext(os.path.basename(arquivo))[0]
        caminho_comum = os.path.join(
            diretorio_processado, f"{nome_dataset}_tradicional_processed.csv")
        caminho_bert = os.path.join(
            diretorio_processado, f"{nome_dataset}_bert_processed.csv")

        # Verificar e carregar versão comum
        if os.path.exists(caminho_comum):
            print(f"Carregando dataset processado (comum): {nome_dataset}")
            dados_comuns = pd.read_csv(caminho_comum)
            lista_datasets_comuns.append(dados_comuns)
        else:
            print(f"Processando dataset bruto (comum): {nome_dataset}")
            dados_filtrados = carregar_dados(arquivo)
            dados_filtrados['dataset_name'] = nome_dataset

            # Aplicar limite, se necessário
            if limitar_registros:
                dados_filtrados = dados_filtrados.head(limite)

            # Preprocessar descrições (comum)
            lista_para_preprocessar = [dados_filtrados]
            dados_processados_comum = preprocessar_todos_datasets(
                lista_para_preprocessar, baseBert=False)[0]

            # Salvar dataset processado (comum)
            dados_processados_comum.to_csv(caminho_comum, index=False)
            print(f"Dataset processado salvo (comum): {caminho_comum}")
            lista_datasets_comuns.append(dados_processados_comum)

        # Verificar e carregar versão BERT
        if os.path.exists(caminho_bert):
            print(f"Carregando dataset processado (BERT): {nome_dataset}")
            dados_bert = pd.read_csv(caminho_bert)
            lista_datasets_bert.append(dados_bert)
        else:
            print(f"Processando dataset bruto (BERT): {nome_dataset}")
            dados_filtrados = carregar_dados(arquivo)
            dados_filtrados['dataset_name'] = nome_dataset

            # Aplicar limite, se necessário
            if limitar_registros:
                dados_filtrados = dados_filtrados.head(limite)

            # Preprocessar descrições (BERT)
            lista_para_preprocessar = [dados_filtrados]
            dados_processados_bert = preprocessar_todos_datasets(
                lista_para_preprocessar, baseBert=True)[0]

            # Salvar dataset processado (BERT)
            dados_processados_bert.to_csv(caminho_bert, index=False)
            print(f"Dataset processado salvo (BERT): {caminho_bert}")
            lista_datasets_bert.append(dados_processados_bert)

    return lista_datasets_comuns, lista_datasets_bert


def remove_invalid_characters(text):
    # Remove caracteres substitutos ou inválidos
    return ''.join(c for c in text if not unicodedata.category(c).startswith('Cs'))


def preprocessar_descricao(list_descricao, baseBert=False):
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

                if not baseBert:
                    # Processa o texto com o spaCy apenas se não for para uso com BERT
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


def preprocessar_todos_datasets(lista_datasets, baseBert=False):
    for dados_filtrados in lista_datasets:
        dados_filtrados['treated_description'] = preprocessar_descricao(
            dados_filtrados['description'].values, baseBert)

    return lista_datasets


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

# Função para gerar as estatísticas da base


def gerar_estatisticas_base(dados_filtrados):
    # Nome da base de dados
    nome_base = dados_filtrados['dataset_name'].iloc[0]

    # Total de registros
    total_registros = len(dados_filtrados)

    # Contagem de palavras por registro usando spaCy
    contagem_palavras = dados_filtrados['description'].apply(
        contar_palavras_spacy)
    media_palavras = np.mean(contagem_palavras)
    desvio_palavras = np.std(contagem_palavras)

    # Distribuição de story points (somente para Fibonacci)
    story_points = dados_filtrados['storypoint'].values
    distribuicao_story_points = Counter(story_points)
    distribuicao_fibonacci = {ponto: distribuicao_story_points[ponto]
                              for ponto in fibonacci_sequence if ponto in distribuicao_story_points}

    # Exibir as estatísticas
    print(f"Base de Dados: {nome_base}")
    print(f"Total de Registros: {total_registros}")
    print(f"Média de Palavras por Registro: {media_palavras:.2f}")
    print(f"Desvio Padrão de Palavras por Registro: {desvio_palavras:.2f}")

    # Verificar se existem valores da sequência de Fibonacci
    if distribuicao_fibonacci:
        print(
            f"Distribuição de Story Points (Fibonacci): {distribuicao_fibonacci}")
    else:
        print("Não há story points seguindo a sequência de Fibonacci.")

    print("\n")
