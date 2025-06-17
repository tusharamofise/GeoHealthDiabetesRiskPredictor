from flask import Flask, render_template, request
from flask_material import Material
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)
Material(app)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/preview')
def preview():
    df = pd.read_csv(r"data/custom_target_accuracy_diabetes_dataset.csv")
    return render_template("preview.html", df_view=df)

@app.route('/analyse', methods=['POST'])
def analyse():
    if request.method == 'POST':
        try:
            # Fetch form data
            Age = request.form['Age']
            BMI = request.form['BMI']
            Exercise = request.form['Exercise']
            Fast_Food_Density = request.form['Fast_Food_Density']
            Green_Space = request.form['Green_Space']
            Air_Quality = request.form['Air_Quality']
            Walkability = request.form['Walkability']
            Family_History = request.form['Family_History']
            model_choice = request.form['model_choice']

            # Convert to float
            sample_data = [Age, BMI, Exercise, Fast_Food_Density, Green_Space, Air_Quality, Walkability, Family_History]
            clean_data = [float(i) for i in sample_data]
            ex1 = np.array(clean_data).reshape(1, -1)

            # Load model and predict
            if model_choice == 'logit_model':
                model = joblib.load("data/logit_model_iris1.pkl")
                model_accuracy = 89
            elif model_choice == 'randomforest_model':
                model = joblib.load("data/randomforest_model_iris1.pkl")
                model_accuracy = 92.0
            elif model_choice == 'svm_model':
                model = joblib.load("data/svm_model_iris1.pkl")
                model_accuracy = 89.5
            else:
                return "Invalid Model Selected"

            result_prediction = model.predict(ex1)

            return render_template("index.html",
                Age=Age,
                BMI=BMI,
                Exercise=Exercise,
                Fast_Food_Density=Fast_Food_Density,
                Green_Space=Green_Space,
                Air_Quality=Air_Quality,
                Walkability=Walkability,
                Family_History=Family_History,
                clean_data=clean_data,
                result_prediction=result_prediction,
                model_selected=model_choice,
                selected_model_accuracy=model_accuracy
            )

        except Exception as e:
            return f"Something went wrong: {e}"

if __name__ == "__main__":
    app.run(debug=True)
