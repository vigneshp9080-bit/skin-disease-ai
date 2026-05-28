from flask import Flask, render_template, request
import numpy as np
import joblib

from feature_names import feature_names

app = Flask(__name__)

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    BASE_DIR,
    '..',
    'models',
    'skin_disease_model.pkl'
)

model = joblib.load(model_path)
# Disease Mapping
disease_classes = {
    1: "Psoriasis",
    2: "Seborrheic Dermatitis",
    3: "Lichen Planus",
    4: "Pityriasis Rosea",
    5: "Chronic Dermatitis",
    6: "Pityriasis Rubra Pilaris"
}


@app.route('/')
def home():

    return render_template(
        'index.html',
        features=feature_names
    )


@app.route('/predict', methods=['POST'])
def predict():

    try:

        features = [
            float(x)
            for x in request.form.values()
        ]

        final_features = np.array(features).reshape(1, -1)

        prediction = model.predict(final_features)

        disease = disease_classes.get(
            prediction[0],
            "Unknown Disease"
        )

        return render_template(
            'index.html',
            prediction_text=f'Predicted Disease: {disease}',
            features=feature_names
        )

    except Exception as e:

        return render_template(
            'index.html',
            prediction_text=f'Error: {str(e)}',
            features=feature_names
        )


if __name__ == "__main__":
    app.run(debug=True)
