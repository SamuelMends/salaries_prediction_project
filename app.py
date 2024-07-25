# Imports

import pickle
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler

# Cria o app
app = Flask(__name__)

# Carregando o modelo e o padronizador:
modelo_carregado = pickle.load(open('salary.model.pkl', 'rb'))
modelo_sam = modelo_carregado["model"]
country_mapping = modelo_carregado["country_mapping"]
education_mapping = modelo_carregado["education_mapping"]
devtype_mapping = modelo_carregado["devtype_mapping"]
scaler_sam = pickle.load(open('dsam_scaler.pkl', 'rb'))

# Rota para a raiz da aplicação
@app.route('/')
def home():
    return render_template('home.html')