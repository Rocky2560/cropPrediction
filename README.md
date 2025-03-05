# 🌱 Crop Recommendation System

## 📌 Overview
This project consists of a **machine learning model** trained using a **Random Forest Classifier** to recommend suitable crops based on environmental factors like **temperature and humidity**. Additionally, a **Flask web application** fetches real-time weather data using the OpenWeather API and provides crop recommendations accordingly.

---
## 📌 Demo
https://cropprediction-trjp.onrender.com/
---
## 📂 Project Structure
```
├── Crop_recommendation.csv   # Dataset used for training
├── crop_model.pkl            # Saved ML model
├── train_model.py            # Script to train and save the model
├── app.py                    # Flask web application
├── templates/
│   ├── index.html            # Frontend for the web app
├── README.md                 # Project documentation
```

---

## ⚙️ Setup Instructions

### 1️⃣ Install Dependencies
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 2️⃣ Train the Model (Optional)
If needed, re-train the model using:
```bash
python train_model.py
```

### 3️⃣ Run the Flask App
```bash
python app.py
```
Then, open `http://127.0.0.1:5000/` in your browser.

---

## 🚀 How It Works
1. Enter a city name in the web app.
2. The app fetches **real-time temperature & humidity** from OpenWeather API.
3. The **ML model predicts** a suitable crop for the conditions.
4. The result is displayed on the webpage.

---

## 🔧 Environment Variables
Set your OpenWeather API key as an environment variable:
```bash
export OPENWEATHER_API_KEY='your_api_key_here'
```

---

## 📌 Future Improvements
- Add more **features** like soil nutrients and rainfall.
- Improve model accuracy using **hyperparameter tuning**.
- Deploy the app using **Docker, AWS, or Railway.app**.

---

## 📜 License
This project is **open-source**. Feel free to modify and improve! 🚀
