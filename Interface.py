import pandas as pd
import streamlit as st
import joblib

# Separando as features do dataframe em categorias para melhor serem trabalhadas
# categorias numéricas
x_numericos = {'age': 0, 'chest pain type': 0, 'resting bp s': 0, 'cholesterol': 0, 'fasting blood sugar': 0, 'resting ecg': 0,
               'max heart rate': 0, 'oldpeak': 0, 'ST slope': 0, }

# categorias com se tem listas de tópicos
x_listas = {
    'sex': ['male', 'female'],
    'exercise angina': ['yes', 'no']
    }

dicionario = {}
for item in x_listas:
    for valor in x_listas[item]:
        dicionario[f'{item}_{valor}'] = 0

# criando os botoes para valores numéricos
for item in x_numericos:
    valor = st.number_input(f'{item}', step=1, value=0)
    x_numericos[item] = valor

# Criando botoes onde se terá uma lista com as categorias
for item in x_listas:
    valor = st.selectbox(f'{item}', x_listas[item])
    dicionario[f'{item}_{valor}'] = 1

botao = st.button('Situação Paciente')
if botao:
    dicionario.update(x_numericos)  # junta o x_numericos ao dicionario onde já tem o x_listas
    x_valores = pd.DataFrame(dicionario, index=[0])  # transforma o dicionario em um DF com indice 0
    modelo = joblib.load('modelo.joiblib')  # carrega o modelo de ML com extensão joblib ao deploy
    preco = modelo.predict(x_valores)  # realiza a predição no modelo com o DF de x_valores
    st.write(preco[0])  # retorna o valor predito pelo modelo