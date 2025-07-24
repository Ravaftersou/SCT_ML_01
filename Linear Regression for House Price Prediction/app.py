from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sqft = float(request.form['square_feet'])
    beds = int(request.form['bedrooms'])
    baths = int(request.form['bathrooms'])

    input_data = np.array([[sqft, beds, baths]])
    prediction = model.predict(input_data)

    # Get coefficients
    intercept = model.intercept_
    coef = model.coef_
    features = ['Square Feet', 'Bedrooms', 'Bathrooms']
    coefficients = zip(features, coef)

    return render_template(
        'index.html',
        prediction_text=f'Estimated Price: ₹{prediction[0]:,.2f}',
        intercept=intercept,
        coefficients=coefficients
    )

if __name__ == "__main__":
    app.run(debug=True)
