import pandas as pd
import numpy as np
from base_am.preprocessamento_atributos import BagOfWords, BagOfItems




def gerar_atributos_ator(df_treino: pd.DataFrame, df_data_to_predict: pd.DataFrame) -> pd.DataFrame:
    obj_bag_of_actors = BagOfItems(min_occur=3)
    df_treino_boa = obj_bag_of_actors.cria_bag_of_items(
        df_treino, ["ator_1", "ator_2", "ator_3", "ator_4", "ator_5"])
    df_data_to_predict_boa = obj_bag_of_actors.aplica_bag_of_items(
        df_data_to_predict, ["ator_1", "ator_2", "ator_3", "ator_4", "ator_5"])

    return df_treino_boa, df_data_to_predict_boa


def gerar_atributos_resumo(df_treino: pd.DataFrame, df_data_to_predict: pd.DataFrame) -> pd.DataFrame:
    bow_amostra = BagOfWords()
    df_bow_treino = bow_amostra.cria_bow(df_treino, "resumo")
    df_bow_data_to_predict = bow_amostra.aplica_bow(
        df_data_to_predict, "resumo")

    return df_bow_treino, df_bow_data_to_predict


def cria_dm(df_data: pd.DataFrame, coluna: str):
    for i, valor in enumerate(df_data[coluna]):
        if type(valor) != str:
            idx_item = df_data[coluna].index[i]
            df_data.at[idx_item, coluna] = 0
        else:
            idx_item = df_data[coluna].index[i]
            df_data.at[idx_item, coluna] = int(valor[0:4])
    return df_data

# Para os nossos experimentos criamos uma função para gerar os atributos com os elementos que acreditamos ser relevantes



def gerar_atributos(df: pd.DataFrame, bag_cast: BagOfItems) -> pd.DataFrame:

    # cria uma sacola de itens com os elencos mais famosos
    df_bag_cast = bag_cast.cria_bag_of_items(
        df, ["ator_1", "ator_2", "ator_3", "ator_4", "ator_5", "dirigido_por"])

    # mapeamento do genero
    if "genero" in df:
        dic_genero = {'Comedy': 1, 'Action': 2}
        for i, valor in enumerate(df["genero"]):
            valor_int = dic_genero[valor]
            df["genero"].iat[i] = valor_int
        df_genero = df.filter(items=["genero"])

    # cria dataframe com as datas
    df_bag_dates = cria_dm(df_data=df, coluna="data_de_estreia").filter(items=["data_de_estreia"])
            
    
    # agora o código ira concatenar os dataframes que foram modificados
    if "genero" in df:  # caso não exista a coluna genero (df de testes)
        df_preprocessado = pd.concat(
            [df_bag_cast, df_bag_dates,df_genero], axis=1
        )
    else:
        df_preprocessado = pd.concat(
            [df_bag_cast, df_bag_dates], axis=1
        )

    df_preprocessado = df_preprocessado.T.drop_duplicates().T
    df_preprocessado.set_index("id")
    return df_preprocessado

# gerar o dataframe de testes com os atributos do dataframe usado no treino sem a col_classe


def gerar_atributos_teste(df_treino: pd.DataFrame, df_teste: pd.DataFrame, col_classe: str):
    # salva as colunas do dataframe de treino
    columns = df_treino.drop("genero",axis=1).columns

    # preprocessa o dataframe de teste
    df_preprocessado_teste = gerar_atributos(df_teste, BagOfItems(0))

    #cria um dataframe em branco com a quantidade linhas igual ao dataframe de teste e as col
    df = pd.DataFrame(columns=columns, data=np.zeros((len(df_teste.index), len(columns.array) ) ) )
    
    for column in df_preprocessado_teste.columns:
        if column in columns:
            df[column] = df_preprocessado_teste[column]

    return df
