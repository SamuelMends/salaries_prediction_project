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

# Rota para a API de Previsão:
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {
            'Country': request.form['Country'],
            'Education': request.form['education'],
            'DevType': request.form['devtype'],
            'Experience': float(request.form['experience']),
    }
    except KeyError as e:
        return render_template("home.html", prediction_text = f"Entrada inválida. Erro: {e}")
    
    # Verifica se algum campo está vazio:
    if any(value == '' for value in data.values()):
        return render_template("home.html", prediction_text = "Verifique se todos os campos estão preenchidos.")
    
    # Aplica o padronizador
    dados_padronizados = scaler_sam.transform([list(data.values())])
    
    # Previsão com o modelo:
    output = modelo_sam.predict(dados_padronizados)[0]
    
    # Formata a saída
    formatted_output = round(output, 2)
    
    # Renderiza o html com a previsão do modelo
    return render_template("home.html", prediction_text = f"$ {format(formatted_output)} Per year")

# Executa o app:
if __name__ == "__main__":
    app.run()