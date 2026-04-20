from flask import Flask, request, jsonify
from flask_cors import CORS
from flasgger import Swagger, swag_from
import numpy as np
import pandas as pd
import joblib
import logging
from datetime import datetime

# Load model dan preprocessor
model = joblib.load('svm_ckd_pipeline.pkl')
preprocessor = joblib.load('selected_features .pkl')

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
                    'packed_cell_volume': {'type': 'number', 'example': 40},
                    'specific_gravity': {'type': 'number', 'example': 1.02},
                    'serum_creatinine': {'type': 'number', 'example': 1.2},
                    'appetite': {'type': 'string', 'example': 'good'},
                    'hypertension': {'type': 'string', 'example': 'no'},
                    'blood_urea': {'type': 'number', 'example': 40},
                    'diabetes_mellitus': {'type': 'string', 'example': 'no'},
                    'sodium': {'type': 'number', 'example': 140},
                    'albumin': {'type': 'number', 'example': 0},
                    'red_blood_cell_count': {'type': 'number', 'example': 5.0},
                    'aanemia': {'type': 'string', 'example': 'no'}
                },
                'required': [
                    'haemoglobin', 'packed_cell_volume', 'specific_gravity', 'serum_creatinine',
                    'appetite', 'hypertension', 'blood_urea', 'diabetes_mellitus', 
                    'sodium', 'albumin', 'red_blood_cell_count', 'aanemia'
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
                    'timestamp': {'type': 'string', 'example': '2025-06-15T12:34:56.789123'},
                    'note': {'type': 'string', 'example': 'Hasil ini merupakan prediksi berdasarkan model machine learning.Jika ingin memastikan keakuratan prediksi, segera periksa kefasilitas kesehatan (Faskes) terdekat untuk konsultasi dokter.'}
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
            float(data['packed_cell_volume']),
            float(data['specific_gravity']),
            float(data['serum_creatinine']),
            data['appetite'],
            data['hypertension'],
            float(data['blood_urea']),
            data['diabetes_mellitus'],
            float(data['sodium']),
            float(data['albumin']),
            float(data['red_blood_cell_count']),
            data['aanemia']

        ]]).reshape(1, -1)

        features = ['haemoglobin', 'packed_cell_volume', 'specific_gravity', 'serum_creatinine', 'appetite', 'hypertension', 'blood_urea', 'diabetes_mellitus', 'sodium', 'albumin', 'red_blood_cell_count', 'aanemia']
        
        input_df = pd.DataFrame(values, columns=features)
        predicted = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        prob_dict = {
            label_classes[i]: float(probabilities[i]) for i in range(len(label_classes))
        }
        logging.info(f"Probabilities: {prob_dict}")
        # Return response
        return jsonify({
            "success": True,
            "predicted_label": label_classes[predicted],
            "probabilities": prob_dict,
            "timestamp": datetime.now().isoformat(),
            "note":"Hasil ini merupakan prediksi berdasarkan model machine learning.Jika ingin memastikan keakuratan prediksi, segera periksa kefasilitas kesehatan (Faskes) terdekat untuk konsultasi dokter."
        })

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
