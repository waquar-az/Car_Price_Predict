from flask import Flask,redirect,url_for,render_template,request
from flask_cors import CORS,cross_origin
import pandas as pd
import numpy as np
import pickle

app=Flask(__name__)
model =pickle.load(open("LinearRegression.pkl",'rb'))
car=pd.read_csv('cleaned_Car.csv')


@app.route('/')
def index():
    companies=sorted(car['company'].unique())
    car_models=sorted(car['name'].unique())
    year=sorted(car['year'].unique(),reverse=True)
    fuel_type=car['fuel_type'].unique()
    
    return render_template('index.html',companies=companies, car_models=car_models, years=year,fuel_types=fuel_type)

@app.route('/predict',methods=['POST'])
#@cross_origin()
def predict():
    company=request.form.get('company')



    car_model=request.form.get('car_models')
    year=request.form.get('year')
    fuel_type=request.form.get('fuel_type')
    kms_driven=request.form.get('kilo_driven')

    prediction=model.predict(pd.DataFrame([[car_model, company, year,kms_driven,fuel_type]],
                              columns=['name','company','year','kms_driven','fuel_type']))
    print(prediction)

    return str(np.round(prediction[0],3))

    
if __name__ == '__main__':
  
  
    app.run(port=5000,debug=True)