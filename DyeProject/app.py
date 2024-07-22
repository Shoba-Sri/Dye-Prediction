from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the saved model
model = load_model('trained_model.h5')

# Load the dataset
data = pd.read_excel(r'C:\Users\aswat\Documents\Dye\Dataset\data_both_monoazo.xlsx')

# Extract the features used for scaling
features = ['Conc of H2O2 (mM)','Dose of gamma ray  (kGy)','pH','Concentration of dye (mg/L)']

# Compute the min and max values for scaling
min_vals = data[features].min().values.astype(np.float64)
max_vals = data[features].max().values.astype(np.float64)

# Define the Min-Max scaling function
def min_max_scaling(sample_input):
    return (sample_input - min_vals) / (max_vals - min_vals)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the HTML form
    h2o2_concentration = float(request.form['h2o2_concentration'])
    gamma_ray_dose = float(request.form['gamma_ray_dose'])
    ph_value = float(request.form['ph_value'])
    dye_concentration = float(request.form['dye_concentration'])
    
    # Prepare input data for prediction
    sample_input = np.array([[h2o2_concentration, gamma_ray_dose, ph_value, dye_concentration]])
    sample_input_scaled = min_max_scaling(sample_input)
    
    # Make predictions
    predicted_output = model.predict(sample_input_scaled)
    
    # Format prediction result
    prediction_percentage = round(float(predicted_output[0][0]) * 100, 2)
    
    # Send prediction result back to the HTML page
    return render_template('index.html', prediction=prediction_percentage)

if __name__ == '__main__':
    app.run(debug=True)
