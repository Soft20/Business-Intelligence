from joblib import load
from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)
model = load('stress-predictor.joblib')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    age = request.form.get('age')
    gender = request.form.get('gender')
    self_employed = request.form.get('self_employed')
    family_history = request.form.get('family_history')
    remote_work = request.form.get('remote_work')
    tech_company = request.form.get('tech_company')
    benefits = request.form.get('benefits')
    care_options = request.form.get('care_options')
    wellness_program = request.form.get('wellness_program')

    prediction = SuggestHelp(age, gender, self_employed, family_history, remote_work, tech_company, benefits, care_options,
                             wellness_program)

    # SuggestHelp(50, 'female', 'yes', 'no', 'yes', 'yes', 'no', 'no', 'no')

    return render_template('prediction.html', prediction=prediction)


def SuggestHelp(Age, Gender, self_employed, family_history, remote_work, tech_company, benefits, care_options, wellness_program):

    Gender = 0 if Gender == 'male' else 1
    self_employed = 1 if self_employed else 0
    family_history = 1 if family_history else 0
    remote_work = 1 if remote_work else 0
    tech_company = 1 if tech_company else 0
    benefits = 1 if benefits else 0
    care_options = 1 if care_options else 0
    wellness_program = 1 if wellness_program else 0

    X = np.column_stack([Age, Gender, self_employed, family_history, remote_work, tech_company, benefits, care_options, wellness_program]).astype('int32')

    prediction = model.predict(X)[0]
    print('prediction::', prediction)

    return 'You got the stress' if round(prediction) == 1 else 'no worries, be happy'


if __name__ == '__main__':
    app.run(port=5000)


'''
Age,
Gender
, self_employed
, family_history
, remote_work
, tech_company
, benefits
, care_options
, wellness_program):
'''
