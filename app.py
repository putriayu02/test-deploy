from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from google.cloud import storage
import firebase_admin
from firebase_admin import credentials, firestore
import os
from PIL import Image
import io
import uuid
from datetime import datetime

# Inisialisasi Firebase Admin
cred = credentials.Certificate("service-account.json")  # Ganti dengan path file JSON Anda
firebase_admin.initialize_app(cred)
db = firestore.client()

# Inisialisasi Flask App
app = Flask(__name__)

# Google Cloud Storage Bucket Info
BUCKET_NAME = "storage-model-data"  # Ganti dengan nama bucket Anda
MODEL_PATH_IN_BUCKET = "model_capstonelancar.h5"  # Path ke model dalam bucket

# Fungsi untuk mengunduh model dari bucket
def download_model_from_gcs(bucket_name, model_path_in_bucket):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(model_path_in_bucket)
    local_model_path = "/tmp/model.h5"  # Path sementara di container Docker
    blob.download_to_filename(local_model_path)
    return local_model_path

# Load model dari GCS saat aplikasi dimulai
print("Mengunduh model dari bucket...")
model_path = download_model_from_gcs(BUCKET_NAME, MODEL_PATH_IN_BUCKET)
model = tf.keras.models.load_model(model_path)
print("Model berhasil dimuat.")

# Label klasifikasi
class_labels = ['battery', 'biological', 'brown-glass', 'cardboard', 
                'clothes', 'green-glass', 'metal', 'paper', 
                'plastic', 'shoes', 'trash', 'white-glass']

# Fungsi untuk memproses gambar dan membuat prediksi
def process_image(image):
    # Ubah gambar menjadi array NumPy untuk model TensorFlow
    image = image.resize((224, 224))  # Resize sesuai dengan dimensi input model
    image_array = np.array(image) / 255.0  # Normalisasi gambar
    image_array = np.expand_dims(image_array, axis=0)  # Tambahkan batch dimension
    return image_array

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil file gambar dari request
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400

    try:
        # Baca gambar dan konversi ke format yang sesuai
        image = Image.open(io.BytesIO(file.read()))

        # Proses gambar
        input_data = process_image(image)
        
        # Prediksi
        predictions = model.predict(input_data)
        predicted_class = class_labels[np.argmax(predictions)]

        # Generate unique ID untuk setiap prediksi
        prediction_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()

        # Simpan hasil prediksi ke Firestore
        doc_ref = db.collection("predictions").document(prediction_id)
        doc_ref.set({
            "id": prediction_id,
            "result": predicted_class,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "createdAt": created_at
        })
        
        # Respons yang diinginkan
        return jsonify({
            "status": "success",
            "message": "Model is predicted successfully",
            "data": {
                "id": prediction_id,
                "result": predicted_class,
                "createdAt": created_at
            }
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Endpoint untuk mendapatkan riwayat prediksi
@app.route('/predict-history', methods=['GET'])
def get_predict_history():
    try:
        # Ambil semua data prediksi dari Firestore
        predictions_ref = db.collection("predictions")
        predictions = predictions_ref.stream()
        
        # Mengumpulkan hasil prediksi
        prediction_history = []
        for prediction in predictions:
            prediction_data = prediction.to_dict()
            prediction_history.append({
                "id": prediction_data.get("id"),
                "result": prediction_data.get("result"),
                "createdAt": prediction_data.get("createdAt")
            })
        
        # Respons yang diinginkan
        return jsonify({
            "status": "success",
            "data": prediction_history
        })
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)  # Port 8080 untuk Cloud Run
