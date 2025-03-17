from preprocessing import carregar_datasets_unificado
from ml_evaluation import avaliar_modelos_intra_datasets, avaliar_modelos_inter_datasets


if __name__ == '__main__':

    versao_abordagem = 'V_ENSEMBLE'

    diretorio_datasets_brutos = 'datasets_all/'
    diretorio_datasets_processados = 'datasets_all_processados/'

    limitar_qtde_registros = False
    n_registros = 10

    is_run_intra_datasets = False

    nome_arquivo_resultados = 'resultados_modelos.csv'

    print('\nCarregando Bases de Dados')

    datasets = carregar_datasets_unificado(
        diretorio_datasets_brutos,
        diretorio_datasets_processados,
        limitar_qtde_registros,
        n_registros)

    print('\n\n')

    if is_run_intra_datasets:

        avaliar_modelos_intra_datasets(
            lista_datasets=datasets,
            versao_nome=versao_abordagem,
            nome_arquivo_csv=nome_arquivo_resultados
        )

    else:

        avaliar_modelos_inter_datasets(
            lista_datasets=datasets,
            versao_nome=versao_abordagem,
            nome_arquivo_csv='resultados_modelos_inter_dataset.csv'
        )
