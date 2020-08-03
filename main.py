from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd

from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


@app.route('/', methods=['GET'])
@cross_origin()
def homepage():
    return render_template("index.html")


@app.route('/predict', methods=['GET', 'POST'])
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            pregnancies = float(request.form['Pregnancies'])
            glucose = float(request.form['Glucose'])
            bloodpressure = float(request.form['BloodPressure'])
            skinthickness = float(request.form['SkinThickness'])
            insulin = float(request.form['Insulin'])
            bmi = float(request.form['BMI'])
            diabetespedigreefunction = float(request.form['DiabetesPedigreeFunction'])
            age = float(request.form['Age'])

            filename1 = 'modelForPrediction.sav'
            loaded_model = pickle.load(open(filename1, 'rb'))
            filename2 = 'logistic_stdscaler.sav'
            scaler_model = pickle.load(open(filename2, 'rb'))

            input_data = [[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigreefunction, age]]
            input_df = pd.DataFrame(input_data)
            scaled_input_df = scaler_model.transform(input_df)
            prediction = loaded_model.predict(scaled_input_df)
            if prediction == 1:
                result = 'Diabetic'
            else:
                result = 'Non-Diabetic'

            print("Prediction is", result)
            return render_template('results.html', prediction=result)
        except Exception as e:
            print('The Exception  message is:', e)
            return 'Something is Wrong'
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5001, debug=True)
#   app.run(debug=True)  running the app
