from flask import Flask, request, jsonify
from flask_cors import CORS
from flasgger import Swagger, swag_from
import numpy as np
import pandas as pd
import joblib
import logging
from datetime import datetime

# Load model dan preprocessor
model = joblib.load('preprocessor_fitur.pkl')
preprocessor = joblib.load('preprocessor_fitur.pkl')

# Label klasifikasi (pastikan urutannya sesuai dengan label pada model)
label_classes = ['Negatif CKD', 'Positif CKD']

# Setup logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Init Flask app
app = Flask(__name__)
CORS(app)
swagger = Swagger(app)

@app.route('/')
def home():
    return jsonify({"message": "Test API"})

@app.route('/predict', methods=['POST'])
@swag_from({
    'tags': ['Prediction'],
    'summary': 'Predict Kidney Disease Classification',
    'description': 'Menerima input fitur numerik dan mengembalikan hasil klasifikasi penyakit ginjal.',
    'consumes': ['application/json'],
    'produces': ['application/json'],
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'haemoglobin': {'type': 'number', 'example': 12.3},
                    'specific_gravity': {'type': 'number', 'example': 1.02},
                    'albumin': {'type': 'number', 'example': 0},
                    'blood_glucose_random': {'type': 'number', 'example': 110},
                    'sugar': {'type': 'number', 'example': 0},
                    'potassium': {'type': 'number', 'example': 45},
                    'blood_urea': {'type': 'number', 'example': 40},
                    'blood_pressure': {'type': 'number', 'example': 80},
                    'serum_creatinine': {'type': 'number', 'example': 1.2},
                    'sodium': {'type': 'number', 'example': 140}
                },
                'required': [
                    'haemoglobin', 'specific_gravity', 'albumin', 'blood_glucose_random',
                    'sugar', 'potassium', 'blood_urea', 'blood_pressure', 
                    'serum_creatinine', 'sodium'
                ]
            }
        }
    ],
    'responses': {
        200: {
            'description': 'Prediction result',
            'schema': {
                'type': 'object',
                'properties': {
                    'success': {'type': 'boolean', 'example': True},
                    'predicted_label': {'type': 'string', 'example': 'Positif CKD'},
                    'probabilities': {
                        'type': 'object',
                        'properties': {
                            'Negatif CKD': {'type': 'number', 'example': 0.1345},
                            'Positif CKD': {'type': 'number', 'example': 0.8655}
                        }
                    },
                    'timestamp': {'type': 'string', 'example': '2025-06-15T12:34:56.789123'}
                }
            }
        },
        500: {
            'description': 'Internal server error',
            'schema': {
                'type': 'object',
                'properties': {
                    'success': {'type': 'boolean', 'example': False},
                    'error': {'type': 'string', 'example': 'Internal server error'}
                }
            }
        }
    }
})
def predict():
    try:
        data = request.json
        logging.info(f"Received prediction request: {data}")

        # Masukkan data ke DataFrame dengan urutan yang sesuai
        values = np.array([[
            float(data['haemoglobin']),
            float(data['blood_glucose_random']),    
            float(data['blood_urea']),
            float(data['potassium']),
            float(data['sodium']),
            float(data['serum_creatinine']),
            float(data['specific_gravity']),
            float(data['blood_pressure']),
            float(data['albumin']),
            float(data['sugar'])





        ]]).reshape(1, -1)

        features = ['haemoglobin', 'blood_glucose_random', 'blood_urea','potassium', 'sodium', 'serum_creatinine','specific_gravity', 'blood_pressure', 'albumin', 'sugar']
        
        input_df = pd.DataFrame(values, columns=features)
        df_processed = preprocessor.transform(input_df)
        predicted = model.predict(df_processed)[0]
        probabilities = model.predict_proba(df_processed)[0]
        prob_dict = {
            label_classes[i]: float(probabilities[i]) for i in range(len(label_classes))
        }
        logging.info(f"Probabilities: {prob_dict}")
        # Return response
        return jsonify({
            "success": True,
            "predicted_label": label_classes[predicted],
            "probabilities": prob_dict,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
