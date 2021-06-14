from base_am.avaliacao import OtimizacaoObjetivo
from base_am.metodo import MetodoAprendizadoDeMaquina
from base_am.resultado import Fold, Resultado
from competicao_am.metodo_competicao import MetodoCompeticao
import optuna
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier



class OtimizacaoObjetivoSVMCompeticao(OtimizacaoObjetivo):
    def __init__(self, fold:Fold, num_arvores_max:int=5):
        super().__init__(fold)
        self.num_arvores_max = num_arvores_max

    def obtem_metodo(self,trial: optuna.Trial)->MetodoAprendizadoDeMaquina:
        #Um custo adequado para custo pode variar muito, por ex, para uma tarefa 
        #o valor de custo pode ser 10, para outra, 32000. 
        #Assim, normalmente, para conseguir valores mais distintos,
        #usamos c=2^exp_cost
        exp_cost = trial.suggest_uniform('min_samples_split', -5, 5) 
        scikit_method = LinearSVC(C=10**exp_cost, random_state=2)

        return MetodoCompeticao(scikit_method)

    def resultado_metrica_otimizacao(self,resultado: Resultado) -> float:
        return resultado.macro_f1

    
class OtimizacaoObjetivoArvoreDecisao(OtimizacaoObjetivo):
    def __init__(self, fold:Fold):
        super().__init__(fold)

    def obtem_metodo(self,trial: optuna.Trial) -> MetodoAprendizadoDeMaquina:

        min_samples = trial.suggest_uniform('min_samples_split', 0, 0.5)
        clf_dtree = DecisionTreeClassifier(min_samples_split=min_samples,random_state=2)

        return MetodoCompeticao(clf_dtree)

    def resultado_metrica_otimizacao(self,resultado):
        return resultado.macro_f1

    
class OtimizacaoObjetivoRandomForest(OtimizacaoObjetivo):
    def __init__(self, fold:Fold, num_arvores_max:int=5):
        super().__init__(fold)
        self.num_arvores_max = num_arvores_max

    def obtem_metodo(self,trial: optuna.Trial)->MetodoAprendizadoDeMaquina:
        min_samples_split = trial.suggest_uniform('min_samples_split',0,0.5)
        max_features = trial.suggest_uniform('max_features',0,0.5)
        num_arvores = trial.suggest_int('num_arvores',1,100)
        clf_rf = RandomForestClassifier(min_samples_split=min_samples_split,max_features=max_features,n_estimators=num_arvores,random_state=2)

        return MetodoCompeticao(clf_rf)

    def resultado_metrica_otimizacao(self, resultado:Resultado) ->float:
        return resultado.macro_f1
    

class OtimizacaoObjetivoAdaBoost(OtimizacaoObjetivo):
    def __init__(self, fold:Fold, num_arvores_max:int=5):
        super().__init__(fold)
        self.num_arvores_max = num_arvores_max

    def obtem_metodo(self,trial: optuna.Trial)->MetodoAprendizadoDeMaquina:
        
        exp_cost = trial.suggest_uniform('exp_cost', -5, 5) 
        scikit_method = LinearSVC(C=10**2.0019842214593875, random_state=2)
        clf_rf = AdaBoostClassifier(n_estimators=80,learning_rate=1,random_state=2,base_estimator = scikit_method, algorithm='SAMME')

        return MetodoCompeticao(clf_rf)

    def resultado_metrica_otimizacao(self, resultado:Resultado) ->float:
        return resultado.macro_f1
    

    
class OtimizacaoObjetivoGradientBoosting(OtimizacaoObjetivo):
    def __init__(self, fold:Fold, num_arvores_max:int=5):
        super().__init__(fold)
        self.num_arvores_max = num_arvores_max

    def obtem_metodo(self,trial: optuna.Trial)->MetodoAprendizadoDeMaquina:
        n_estimators = trial.suggest_int('n_estimators',1,100)
        min_samples_split = trial.suggest_uniform('min_samples_split',0,0.5)
        max_features = trial.suggest_uniform('max_features',0,0.5)
        clf_rf = GradientBoostingClassifier(n_estimators=n_estimators,min_samples_split=min_samples_split,random_state=2,max_features=max_features, learning_rate = 1)

        return MetodoCompeticao(clf_rf)

    def resultado_metrica_otimizacao(self, resultado:Resultado) ->float:
        return resultado.macro_f1


    
class OtimizacaoObjetivoExtraTree(OtimizacaoObjetivo):
    def __init__(self, fold:Fold, num_arvores_max:int=5):
        super().__init__(fold)
        self.num_arvores_max = num_arvores_max

    def obtem_metodo(self,trial: optuna.Trial)->MetodoAprendizadoDeMaquina:
        min_samples_split = trial.suggest_uniform('min_samples_split',0,0.5)
        max_features = trial.suggest_uniform('max_features',0,0.5)

        clf_rf = ExtraTreeClassifier(splitter="best", min_samples_split=min_samples_split,max_features=max_features,random_state=2)


        return MetodoCompeticao(clf_rf)

    def resultado_metrica_otimizacao(self, resultado:Resultado) ->float:
        return resultado.macro_f1
    
class OtimizacaoObjetivoAdaELinear(OtimizacaoObjetivo):
    def __init__(self, fold:Fold, num_arvores_max:int=5):
        super().__init__(fold)
        self.num_arvores_max = num_arvores_max

    def obtem_metodo(self,trial: optuna.Trial)->MetodoAprendizadoDeMaquina:
        
        exp_cost = trial.suggest_uniform('exp_cost', -5, 5) 
        
        scikit_method = LinearSVC(C=10**exp_cost, random_state=2)
        clf_rf = AdaBoostClassifier(n_estimators=80,learning_rate=1,random_state=2,base_estimator = scikit_method, algorithm='SAMME')

        return MetodoCompeticao([clf_rf,scikit_method])

    def resultado_metrica_otimizacao(self, resultado:Resultado) ->float:
        return resultado.macro_f1
