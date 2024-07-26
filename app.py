# Projeto 1 - Construção de Aplicação Web e Integração com Machine Learning

# Imports
import pickle
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler

# Cria a app
app = Flask(__name__)

app.config['DEBUG'] = True # Ativar modo de depuração

# Carrega o modelo e o padronizador
modelo_carregado = pickle.load(open('salary_model.pkl', 'rb'))
modelo_dsa = modelo_carregado["model"]
country_mapping = modelo_carregado["country_mapping"]
education_mapping = modelo_carregado["education_mapping"]
devtype_mapping  = modelo_carregado['devtype_mapping']
scaler_dsam = pickle.load(open('dsam_scaler.pkl', 'rb'))

# Rota para a raiz
@app.route('/')
def home():
    return render_template('home.html')

# Rota para a API de previsão
@app.route('/predict', methods=['POST'])
def predict():
    try:
         data = {
            'Country': request.form['Country'],
            'education': request.form['education'],
            'devtype' : request.form['devtype'],
            'experience': float(request.form['experience']),
            
    }
    except KeyError as e:
        return render_template("home.html", prediction_text=f"Entrada inválida. Erro: {e}")

    # Verifica se algum campo está vazio
    if any(value == '' for value in data.values()):
        return render_template("home.html", prediction_text="Verifique se todos os campos estão preenchidos.")
    
    # Aplica o padronizador
    dados_padronizados = scaler_dsam.transform([list(data.values())])

    # Previsão com o modelo
    output = modelo_dsa.predict(dados_padronizados)[0]

    # Formata a saída
    formatted_output = round(output, 2)
    
    # Renderiza o html com a previsão do modelo
    return render_template("home.html", prediction_text="$ {} [valor anual]".format(formatted_output))

# Executa a app
if __name__ == "__main__":
    app.run()