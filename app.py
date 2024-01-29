from flask import Flask, request, render_template
from sklearn.preprocessing import PolynomialFeatures
import pickle
import numpy as np

# Charger le modèle
with open('model.pickle', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Obtenir les valeurs du formulaire
    features = [float(x) for x in request.form.values()]
    features = np.array(features).reshape(1, -1)

    # Transformer les caractéristiques avec PolynomialFeatures
    poly = PolynomialFeatures(degree=2)
    features = poly.fit_transform(features)
    
    # Prédire la consommation de CO2
    prediction = model.predict(features)
    
    return render_template('index.html', prediction_text='La consommation estimée de CO2 est de {:.2f} g/km'.format(prediction[0]))

if __name__ == "__main__":
    app.run(debug=True)