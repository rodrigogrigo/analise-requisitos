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

def combinacao_ja_avaliada_inter(nome_arquivo, versao, model, dataset_treino, dataset_validacao, dataset_teste):
    """
    Verifica se a combinação de datasets para treinamento, validação e teste já existe
    no arquivo de resultados.
    """
    if not os.path.exists(nome_arquivo):
        return False

    try:
        df = pd.read_csv(nome_arquivo)
    except Exception as e:
        print(f"Erro ao ler o arquivo {nome_arquivo}: {e}")
        return False

    # Verifica se existe alguma linha com a mesma combinação de datasets, versão e modelo
    cond = (
        (df["Versao"] == versao) &
        (df["Model"] == model) &
        (df["Dataset_Treino"] == dataset_treino) &
        (df["Dataset_Validacao"] == dataset_validacao) &
        (df["Dataset_Teste"] == dataset_teste)
    )
    return cond.any()

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
