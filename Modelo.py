import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

class Modelo:
    def __init__(self, nome_arq, nome_modelo, prediction_value = 'Target_Severity_Score', take_out_values = False):
        # Carrega os dados do arquivo CSV
        self.df = pd.read_csv(nome_arq)

        # Remove colunas que não contribuem para o modelo
        self.df = self.df.drop(['Patient_ID'], axis=1)

        # Mapeando valores não numericos
        self.mapping()

        # Dropando atributos desnecessários
        self.df = self.df.drop(['Gender', 'Country_Region', 'Year'], axis=1)

        # Separa os dados em características (X) e alvo (Y)
        if(take_out_values):
            X = self.df.drop(["Treatment_Cost_USD","Survival_Years","Target_Severity_Score"], axis=1)
        else:
            X = self.df.drop(prediction_value, axis=1)
        Y = self.df[prediction_value]

        # Configura a validação cruzada (5 divisões)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        mse_scores = []
        r2_scores = []
        self.mse_score = float('inf')  # Armazena o menor erro MSE encontrado

        # Loop de validação cruzada
        for train_index, test_index in kf.split(X):
            # Divide os dados em treino e teste
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

            # Escala os dados (necessário para MLP e SVR)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Escolhe o modelo com base na entrada
            if nome_modelo == "MLP":
                model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', max_iter=1000, random_state=42)
            elif nome_modelo == "SVR":
                model = SVR(kernel='linear', C=1.0, epsilon=0.1)
            elif nome_modelo == "DTR":
                model = DecisionTreeRegressor(random_state=42, max_depth=10)
            else:
                print(
                    "Nome do modelo inválido.\n"
                    "- Use 'MLP' para Rede Neural (Multi Layer Perceptron)\n"
                    "- Use 'SVR' para Máquina de Vetor de Suporte (Support Vector Regression)\n"
                    "- Use 'DTR' para Árvore de Decisão (Decision Tree Regressor)"
                )
                return

            # Treina o modelo
            model.fit(X_train_scaled, y_train)

            # Realiza a predição e avalia o desempenho
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            mse_scores.append(mse)
            r2_scores.append(r2)

            # Armazena o melhor modelo (com menor MSE)
            if mse < self.mse_score:
                self.model = model
                self.scaler = scaler
                self.mse_score = mse

        # Exibe resultados de desempenho
        print("MSE (por fold):", mse_scores)
        print("MSE médio:", np.mean(mse_scores))
        print("R² (por fold):", r2_scores)
        print("R² médio:", np.mean(r2_scores))

    def mapping(self):
        # Mapping Gender column: Male = 0, Female = 1
        gender_mapping = {'Male': 0, 'Female': 1, 'Other':2}
        self.df['Gender'] = self.df['Gender'].map(gender_mapping)

        # Mapping Country_Region: Assume you have a few regions like 'Europe', 'Asia', 'Africa', etc.
        # Adjust these mappings based on your data
        country_region_mapping = {
            'Australia': 0,
            'Brazil': 1,
            'Canada': 2,
            'China': 3,
            'Germany': 4,
            'India': 5,
            'Pakistan': 6,
            'Russia': 7,
            'UK': 8,
            'USA': 9
        }
        self.df['Country_Region'] = self.df['Country_Region'].map(country_region_mapping)

        # Mapping Year: If you just want to convert years to categorical values or keep as is
        # Example mapping of years to a range:
        year_mapping = {
            2015: 0,
            2016: 1,
            2017: 2,
            2018: 3,
            2019: 4,
            2020: 5,
            2021: 6,
            2022: 7,
            2023: 8,
            2024: 9
        }  # Adjust based on your data
        self.df['Year'] = self.df['Year'].map(year_mapping)

        # Mapeia estágios do câncer para valores numéricos
        stage_mapping = {
            "Stage 0": 0,
            "Stage I": 1,
            "Stage II": 2,
            "Stage III": 3,
            "Stage IV": 4
        }
        self.df['Cancer_Stage'] = self.df['Cancer_Stage'].map(stage_mapping)

        # Mapeia tipos de câncer para valores numéricos
        type_mapping = {
            'Breast': 0,
            'Cervical': 1,
            'Colon': 2,
            'Leukemia': 3,
            'Liver': 4,
            'Lung': 5,
            'Prostate': 6,
            'Skin': 7
        }
        self.df['Cancer_Type'] = self.df['Cancer_Type'].map(type_mapping)

    def predicao(self, dado):
        """
        Realiza uma predição com o melhor modelo treinado.
        Parâmetro:
            dado (list): uma amostra com as mesmas características de entrada usadas no treino.
        """
        dado = np.array(dado).reshape(1, -1)               # Garante formato 2D
        dado_escalado = self.scaler.transform(dado)        # Aplica o mesmo escalonamento do treino
        pred = self.model.predict(dado_escalado)           # Faz a previsão
        print(pred)