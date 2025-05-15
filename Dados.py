import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

class Dados:
    def __init__(self, nome_arq):
        # Carrega os dados do arquivo CSV
        self.df = pd.read_csv(nome_arq)

        self.df = self.df.drop(['Patient_ID'], axis=1)

        self.already_encoded = False

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

    def correlacao(self, encoding = False):

        if(not self.already_encoded):
            if(encoding):
                self.encoding()
            else:
                self.mapping()

        self.already_encoded = True

        # Calculate correlation matrix
        corr_matrix = self.df.corr()

        # Display the correlation matrix
        print(corr_matrix)

        # Visualize the correlation matrix using a heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()

    def correlacao_par(self, first_feature, second_feature, encoding = False):
        
        if(not self.already_encoded):
            if(encoding):
                self.encoding()
            else:
                self.mapping()

        self.already_encoded = True

        # Step 3: Create interaction feature
        self.df['New_Feature'] = self.df[first_feature] * self.df[second_feature]

        # Calculate correlation matrix
        corr_matrix = self.df.corr()

        # Display the correlation matrix
        print(corr_matrix)

        # Visualize the correlation matrix using a heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()

    def info(self):
        # Basic info
        print("INFO:")
        self.df.info()

        print("DESCRIBE:")
        print(self.df.describe())

    def encoding(self, only_ordinal = False):

        if(not only_ordinal):
            # Encode 'Gender' using OneHot
            encoder_gender = OneHotEncoder(sparse_output=False)
            gender_encoded = encoder_gender.fit_transform(self.df[['Gender']])
            gender_df = pd.DataFrame(gender_encoded, columns=encoder_gender.get_feature_names_out(['Gender']))
            self.df = self.df.drop(columns=['Gender']).reset_index(drop=True)
            self.df = pd.concat([self.df, gender_df], axis=1)

            # Encode 'Country_Region' using OneHot
            encoder_country = OneHotEncoder(sparse_output=False)
            country_encoded = encoder_country.fit_transform(self.df[['Country_Region']])
            country_df = pd.DataFrame(country_encoded, columns=encoder_country.get_feature_names_out(['Country_Region']))
            self.df = self.df.drop(columns=['Country_Region']).reset_index(drop=True)
            self.df = pd.concat([self.df, country_df], axis=1)

            # Encode 'Cancer_Type' using OneHot
            encoder_type = OneHotEncoder(sparse_output=False)
            type_encoded = encoder_type.fit_transform(self.df[['Cancer_Type']])
            type_df = pd.DataFrame(type_encoded, columns=encoder_type.get_feature_names_out(['Cancer_Type']))
            self.df = self.df.drop(columns=['Cancer_Type']).reset_index(drop=True)
            self.df = pd.concat([self.df, type_df], axis=1)
        else:
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

        # Encode 'Year' using Ordinal
        encoder_year = OrdinalEncoder(categories=[[2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]])
        self.df['Year'] = encoder_year.fit_transform(self.df[['Year']])

        # Encode 'Cancer_Stage' using Ordinal
        encoder_stage = OrdinalEncoder(categories=[["Stage 0", "Stage I", "Stage II", "Stage III", "Stage IV"]])
        self.df['Cancer_Stage'] = encoder_stage.fit_transform(self.df[['Cancer_Stage']])

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