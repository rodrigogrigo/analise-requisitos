import os
import pickle
import pandas as pd

from sklearn.model_selection import KFold
import pandas as pd

# Definição da constante para o caminho da pasta de folds
FOLDS_DIRECTORY = "folds"


def salvar_kfold(dataset: pd.DataFrame, dataset_name: str, n_splits: int = 5,
                 shuffle: bool = True, random_state: int = 42) -> None:

    os.makedirs(FOLDS_DIRECTORY, exist_ok=True)  # Cria a pasta se não existir

    # Remove a extensão do arquivo, se houver
    dataset_name = os.path.splitext(dataset_name)[0]

    filename = os.path.join(
        FOLDS_DIRECTORY, f"kfold_indices_{dataset_name}.pkl")

    if not os.path.exists(filename):

        kf = KFold(n_splits=n_splits, shuffle=shuffle,
                   random_state=random_state)

        indices = [(train_index, test_index)
                   for train_index, test_index in kf.split(dataset)]

        # Salva os índices dos folds em um arquivo específico para o dataset
        with open(file=filename, mode='wb') as f:
            pickle.dump(indices, f)
        print(f"KFold indices saved to file for dataset {dataset_name}.")
    else:
        print(
            f"KFold file already exists for dataset {dataset_name}. Skipping creation.")


def carregar_kfold(dataset_name: str):
    filename = os.path.join(
        FOLDS_DIRECTORY, f"kfold_indices_{dataset_name}.pkl")

    with open(filename, 'rb') as f:
        indices = pickle.load(f)
    print(f"KFold indices loaded from file for dataset {dataset_name}.")
    return indices

def criar_folds_para_datasets(lista_datasets: list):
    """
    Cria e salva os folds para todos os datasets fornecidos na lista.
    """
    for dados_filtrados in lista_datasets:
        dataset_name = dados_filtrados['dataset_name'].iloc[0]
        print(f'Criando folds para o dataset: {dataset_name}')
        salvar_kfold(dados_filtrados, dataset_name)

def combinacao_ja_avaliada_inter(nome_arquivo_csv, versao_nome, nome_train, nome_validacao, nome_teste,
                                 model_name):
    """
    Verifica se a combinação de Treinamento, Validação, Teste e Modelo já existe no arquivo CSV.
    Retorna True se existir, False caso contrário.
    """
    try:
        resultados_existentes = pd.read_csv(nome_arquivo_csv)
        condicao = (
            (resultados_existentes['Versao'] == versao_nome) &
            (resultados_existentes['Dataset_Treino'] == nome_train) &
            (resultados_existentes['Dataset_Validacao'] == nome_validacao) &
            (resultados_existentes['Dataset_Teste'] == nome_teste) &
            (resultados_existentes['Model'] == model_name)
        )
        return condicao.any()
    except FileNotFoundError:
        return False

def combinacao_ja_avaliada_intra(nome_arquivo, versao, model, dataset):
    """
    Verifica se já existe uma linha no arquivo de resultados para a combinação de
    Versao, Model e Dataset (utilizado no cenário intra).
    """
    if not os.path.exists(nome_arquivo):
        return False

    try:
        df = pd.read_csv(nome_arquivo)
    except Exception as e:
        print(f"Erro ao ler o arquivo {nome_arquivo}: {e}")
        return False

    cond = (
        (df["Versao"] == versao) &
        (df["Model"] == model) &
        (df["Dataset"] == dataset)
    )
    return cond.any()
