from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model safely
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values
        area = float(request.form['area'])
        bedrooms = int(request.form['bedrooms'])
        age = int(request.form['age'])

        # Prepare input for model
        features = np.array([[area, bedrooms, age]])

        # Prediction
        prediction = model.predict(features)
        output = round(float(prediction[0]), 2)

        return render_template(
            'index.html',
            prediction_text=f'Estimated House Price: ₹ {output}'
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f"Error: {str(e)}"
        )

# IMPORTANT for Render deployment
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)