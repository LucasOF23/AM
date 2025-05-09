import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Dados:
    def __init__(self, nome_arq):
        # Carrega os dados do arquivo CSV
        self.df = pd.read_csv(nome_arq)

        self.df = self.df.drop(['Patient_ID'], axis=1)

    def distribuicao(self):
        # Itera sobre cada coluna
        for col in self.df.columns:
            plt.figure(figsize=(8, 4))
            
            if self.df[col].dtype == 'object' or self.df[col].nunique() < 20:
                # Categórica ou com poucos valores únicos: plotar barras
                self.df[col].value_counts().sort_index().plot(kind='bar')
                plt.title(f"Distribuição de valores - {col}")
                plt.xlabel("Valor")
                plt.ylabel("Quantidade")
            else:
                # Numérica com muitos valores: usar histograma
                self.df[col].plot(kind='hist', bins=20, edgecolor='black')
                plt.title(f"Distribuição em intervalos - {col}")
                plt.xlabel("Intervalo de valores")
                plt.ylabel("Quantidade")

            plt.tight_layout()
            plt.show()

    def correlacao(self):
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

        # Calculate the correlation matrix (Pearson correlation by default)
        corr_matrix = self.df.corr()

        # Display the correlation matrix
        print(corr_matrix)

        # Visualize the correlation matrix using a heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.show()