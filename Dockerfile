# Gunakan tensorflow sebagai base image Python
FROM tensorflow/tensorflow:2.16.1

# Copy file model dan skrip ke dalam container
WORKDIR /app
COPY model_capstonelancar.h5 .
COPY app.py .
COPY service-account.json .

# Install dependensi
RUN pip install --upgrade pip
RUN apt-get update && apt-get remove -y python3-blinker
RUN pip install flask tensorflow==2.16.1
RUN pip install flask tensorflow numpy google-cloud-storage firebase-admin
RUN pip install pillow

#tambahin env
ENV GOOGLE_APPLICATION_CREDENTIALS="service-account.json"

# Jalankan aplikasi Flask
CMD ["python", "app.py"]
