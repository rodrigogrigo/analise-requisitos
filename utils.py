import os
import pickle
from sklearn.model_selection import KFold

# Definição da constante para o caminho da pasta de folds
FOLDS_DIRECTORY = "folds"


def salvar_kfold(dataset, dataset_name, n_splits=5, shuffle=True, random_state=42):
    os.makedirs(FOLDS_DIRECTORY, exist_ok=True)  # Cria a pasta se não existir
    filename = os.path.join(
        FOLDS_DIRECTORY, f"kfold_indices_{dataset_name}.pkl")

    if not os.path.exists(filename):
        kf = KFold(n_splits=n_splits, shuffle=shuffle,
                   random_state=random_state)
        indices = [(train_index, test_index)
                   for train_index, test_index in kf.split(dataset)]

        # Salva os índices dos folds em um arquivo específico para o dataset
        with open(filename, 'wb') as f:
            pickle.dump(indices, f)
        print(f"KFold indices saved to file for dataset {dataset_name}.")
    else:
        print(
            f"KFold file already exists for dataset {dataset_name}. Skipping creation.")


def carregar_kfold(dataset_name):
    filename = os.path.join(
        FOLDS_DIRECTORY, f"kfold_indices_{dataset_name}.pkl")

    with open(filename, 'rb') as f:
        indices = pickle.load(f)
    print(f"KFold indices loaded from file for dataset {dataset_name}.")
    return indices
