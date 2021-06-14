from competicao_am.metodo_competicao import MetodoCompeticao
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import pickle
from typing import Tuple, List
from base_am.avaliacao import *
from base_am.preprocessamento_atributos import *
from competicao_am.preprocessamento_atributos_competicao import gerar_atributos_resumo, gerar_atributos

class Calculos():
    def imprime(m, indice = ["comedy", "action"]):
        print("                     Predito    ")
        for i in range(len(m)):
            print(indice[i] + str(m[i]))


    def somaMatriz(m1, m2):
        m = []
        for i in range(len(m1)):   
            m.append([])
            for j in range(len(m1[0])):
                m[i].append(m1[i][j] + m2[i][j])
        return (m)


    def retornaMatriz_media(vetor_result):
        for i, matriz in enumerate (vetor_result):
            if i==0:
                aux = matriz.mat_confusao
            elif i==1:
                soma = Calculos.somaMatriz(matriz.mat_confusao, aux)
            else:
                soma = Calculos.somaMatriz(matriz.mat_confusao, soma)

        # Tira a média das matrizes
        for i in range(len(soma)):
            for j in range(len(soma)):
                soma[i][j] = soma[i][j]/len(vetor_result)
        return soma

    
    def calcula_indices_importantes(vetor_result):
        acuracia_media: float = 0
        precisao_media: float = 0
        revocacao_media: float = 0
        f1_classe_media: float = 0
            
        for i, matriz in enumerate (vetor_result):
            acuracia_media = matriz.acuracia + acuracia_media
            precisao_media = precisao_media + matriz.precisao
            revocacao_media = revocacao_media + matriz.revocacao
            f1_classe_media = f1_classe_media + matriz.f1_por_classe
            
        acuracia_media = acuracia_media/len(vetor_result)
        precisao_media = precisao_media/len(vetor_result)
        revocacao_media = revocacao_media/len(vetor_result)
        f1_classe_media = f1_classe_media/len(vetor_result)
        
        return(acuracia_media,precisao_media,revocacao_media,f1_classe_media)

#esta função serve para apresentar os resultados obtidos nos experimentos realizados no dataframe com o elenco e data de estreia
def gerar_resultados_df_cast():
    exp_GradientBoostingClassifier = pickle.load( open( "resultados.exp_GradientBoostingClassifier_1.p", "rb" ) )
    exp_random_forest = pickle.load( open( "resultados.exp_random_forest_1.p", "rb" ) )
    exp_ExtraTreeClassifier = pickle.load( open( "resultados.exp_ExtraTreeClassifier_1.p", "rb" ) )
    exp_AdaBoostClassifier = pickle.load( open( "resultados.exp_AdaBoostClassifier_1.p", "rb" ) )
    exp_arvore_decisao = pickle.load( open( "resultados.exp_arvore_decisao_1.p", "rb" ) )
    exp_linear_svc = pickle.load( open( "resultados.exp_linear_svc_1.p", "rb" ) )
  
    # Realiza a avaliação das métricas em cada um dos experimentos para avaliar qual obteve o melhor desempenho
    # Métricas avaliadas: acurácia, precisao média dos folds, revocação, f1 e macro f1
    acuracia_media_Gradient, precisao_media_Gradient, revocacao_media_Gradient, f1_classe_media_Gradient = Calculos.calcula_indices_importantes(exp_GradientBoostingClassifier.resultados)
    acuracia_media_random_forest, precisao_media_random_forest, revocacao_media_random_forest, f1_classe_media_random_forest = Calculos.calcula_indices_importantes(exp_random_forest.resultados)
    acuracia_media_ExtraTree, precisao_media_ExtraTree, revocacao_media_ExtraTree, f1_classe_media_ExtraTree = Calculos.calcula_indices_importantes(exp_ExtraTreeClassifier.resultados)
    acuracia_media_AdaBoost, precisao_media_AdaBoost, revocacao_media_AdaBoost, f1_classe_media_AdaBoost = Calculos.calcula_indices_importantes(exp_AdaBoostClassifier.resultados)
    acuracia_media_arvore_decisao, precisao_media_arvore_decisao, revocacao_media_arvore_decisao, f1_classe_media_arvore_decisao = Calculos.calcula_indices_importantes(exp_arvore_decisao.resultados)
    acuracia_media_linear_svc, precisao_media_linear_svc, revocacao_media_linear_svc, f1_classe_media_linear_svc = Calculos.calcula_indices_importantes(exp_linear_svc.resultados)

    df_Metricas = pd.DataFrame({'Métricas' : ["acuracia_media", "precisao_media", "revocacao_media", "f1_por_classe_media", "Macro F1"]})
    df_Gradient = pd.DataFrame({'GradientBoostingClassifier' : [acuracia_media_Gradient, precisao_media_Gradient, revocacao_media_Gradient, f1_classe_media_Gradient, exp_GradientBoostingClassifier.macro_f1_avg]})
    df_Random = pd.DataFrame({'RandomForest' : [acuracia_media_random_forest, precisao_media_random_forest, revocacao_media_random_forest, f1_classe_media_random_forest, exp_random_forest.macro_f1_avg]})
    df_ExtraTree = pd.DataFrame({'ExtraTreeClassifier' : [acuracia_media_ExtraTree, precisao_media_ExtraTree, revocacao_media_ExtraTree, f1_classe_media_ExtraTree, exp_ExtraTreeClassifier.macro_f1_avg]})
    df_AdaBoost = pd.DataFrame({'AdaBoostClassifier' : [acuracia_media_AdaBoost, precisao_media_AdaBoost, revocacao_media_AdaBoost, f1_classe_media_AdaBoost, exp_AdaBoostClassifier.macro_f1_avg]})
    df_DecisionTree = pd.DataFrame({'DecisionTreeClassifier' : [acuracia_media_arvore_decisao, precisao_media_arvore_decisao, revocacao_media_arvore_decisao, f1_classe_media_arvore_decisao, exp_arvore_decisao.macro_f1_avg]})
    df_Linear = pd.DataFrame({'Linear_SVC' : [acuracia_media_linear_svc, precisao_media_linear_svc, revocacao_media_linear_svc, f1_classe_media_linear_svc, exp_linear_svc.macro_f1_avg]})

#Concatena os Dataframes para visualização
# Primeiro é mostrada a classe Comedy e depois a classe Action para as mátricas avaliadas por classe
    df_visualizacao = pd.concat([df_Metricas, df_Gradient, df_Random, df_ExtraTree, df_AdaBoost, df_DecisionTree, df_Linear], axis=1)
    return df_visualizacao

#a função apresenta os resultados obtidos através do dataframe de resumos
def gerar_resultados_df_resumo():
    exp_AdaBoostClassifier = pickle.load( open( "resultados.exp_AdaBoostClassifier_resumo_1.p", "rb" ) )
    exp_linear_svc = pickle.load( open( "resultados.exp_linear_svc_resumo_1.p", "rb" ) )

    # Realiza a avaliação das métricas em cada um dos experimentos para avaliar qual obteve o melhor desempenho
    # Métricas avaliadas: acurácia, precisao média dos folds, revocação, f1 e macro f1
    acuracia_media_AdaBoost, precisao_media_AdaBoost, revocacao_media_AdaBoost, f1_classe_media_AdaBoost = Calculos.calcula_indices_importantes(exp_AdaBoostClassifier.resultados)
    acuracia_media_linear_svc, precisao_media_linear_svc, revocacao_media_linear_svc, f1_classe_media_linear_svc = Calculos.calcula_indices_importantes(exp_linear_svc.resultados)

    df = pd.DataFrame({
    'Métricas' : ["acuracia_media_Folds", "precisao_media_Folds", "revocacao_media_Folds", "f1_por_classe_media_Folds", "Macro F1"],
    'AdaBoostClassifier' : [acuracia_media_AdaBoost, precisao_media_AdaBoost, revocacao_media_AdaBoost, f1_classe_media_AdaBoost, exp_AdaBoostClassifier.macro_f1_avg],
    'Linear_SVC' : [acuracia_media_linear_svc, precisao_media_linear_svc, revocacao_media_linear_svc, f1_classe_media_linear_svc, exp_linear_svc.macro_f1_avg]
    })
    return df


def gerar_saida_teste( df_data_to_predict, col_classe, num_grupo):
    """
    Assim como os demais códigos da pasta "competicao_am", esta função 
    só poderá ser modificada na fase de geração da solução. 
    """
    
    #o treino será sempre o dataset completo - sem nenhum dado a mais e sem nenhum preprocessamento
    #esta função que deve encarregar de fazer o preprocessamento
    df_treino = pd.read_csv("datasets/movies_amostra.csv")
    df_treino_resumo, df_teste_resumo = gerar_atributos_resumo(df_treino,df_data_to_predict)
    
    ml_method = MetodoCompeticao(AdaBoostClassifier(n_estimators= 80, learning_rate = 1, base_estimator= LinearSVC(C=10**2.0019842214593875,random_state=2),algorithm='SAMME',random_state=2))
    
    #faça a mesma separação que fizemos em x_treino e y_treino nos dados a serem previstos
    x_treino = df_treino_resumo
    y_treino= gerar_atributos(df_treino, BagOfItems(2))[col_classe].astype(int)
        

    #execute o método fit  de ml_method e crie o modelo
    model = ml_method.ml_method.fit(x_treino,y_treino)

    #retorne o resultado por meio do método predict
    y_predictions = model.predict(df_teste_resumo)
    dic_genero= {'comedy':1,'action':2}    
    
    y_predictions_genre = []
    for y in y_predictions:
        for key,value in dic_genero.items():
            if value == y:
                y_predictions_genre.append(key)
    
    #combina as duas
    #arr_final_predictions = ml_method.combine_predictions(arr_predictions_ator, arr_predictions_bow)
    
    #grava o resultado obtido
    with open(f"predict_grupo_{num_grupo}.txt","w") as file_predict:
        for predict in y_predictions_genre: 
            file_predict.write(predict+"\n")
