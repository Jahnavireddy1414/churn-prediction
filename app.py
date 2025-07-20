import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# Load model and feature columns
model = pickle.load(open('model.sav', 'rb'))
model_columns = pickle.load(open('model_columns.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html', form_data={})  # send empty form_data on initial load

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data
        form_data = request.form.to_dict()

        # Convert numeric fields
        numeric_fields = ['tenure', 'MonthlyCharges', 'TotalCharges']
        for field in numeric_fields:
            form_data[field] = float(form_data.get(field, 0))

        # Create a DataFrame from input
        df_input = pd.DataFrame([form_data])

        # Convert categorical variables to dummies
        df_input = pd.get_dummies(df_input)

        # Reindex to match training columns
        df_input = df_input.reindex(columns=model_columns, fill_value=0)

        # Predict
        prediction = model.predict(df_input)[0]
        result = "Churn" if prediction == 1 else "No Churn"

        return render_template('home.html', prediction=result, form_data=request.form)

    except Exception as e:
        return render_template('home.html', prediction=f"Error: {str(e)}", form_data=request.form)

if __name__ == '__main__':
    app.run(debug=True)
