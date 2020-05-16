from flask import Flask, request
import pandas as pd
import pickle
import os

from wine_quality.WineQuality import WineQuality

# carregando o modelo na memória para agilidade do processo
model = pickle.load(open('model/model_wine_quality.pkl', 'rb'))

# instanciar o flask com uma variável app
app = Flask( __name__ )

# criando um end point para previsao dos dados
@app.route('/predict', methods=['POST'])
def predict():
	test_json = request.get_json()

	# coletando os dados
	if test_json:
		# verifica e caso, os valores recebidos são únicos faça assim
		if isinstance(test_json, dict):
			df_raw = pd.DataFrame(test_json, index[0])
		# verifica e caso, os valores são múltiplos é assim
		else:
			df_raw = pd.DataFrame(test_json, columns=test_json[0].keys())

	# instanciando a preparação dos dados
	pipeline = WineQuality()

	# preparação dos dados
	df1 = pipeline.data_preparation( df_raw )

	# Previsão
	pred = model.predict(df1)

	# response
	df1['prediction'] = pred # colocando as previsões em uma coluna do dataframe

	# retornar as previsões em json
	return df1.to_json(orient='records') # "records" é a orientação em que os dados chegaram, então retorno no formato


if __name__ == '__main__':

	# iniciando o flask
	port = int(os.environ.get("PORT", 5000))
	app.run(host='127.0.0.1', port=port, debug=True)
