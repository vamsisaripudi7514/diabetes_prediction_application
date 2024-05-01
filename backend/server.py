from flask import Flask,request,jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS
app = Flask(__name__)
CORS(app,resources={r'/*':{'origins':'*'}})

data = pd.read_csv(r'.\backend\diabetes.csv')


def Manual(data, numeric_columns=None, outlier_factor=1.5, keep_outliers=False):
    data_copy = data.copy()

    if numeric_columns is None:
        numeric_columns = data_copy.select_dtypes(include=['number']).columns

    for col in numeric_columns:
        Q1 = data_copy[col].quantile(0.25)
        Q3 = data_copy[col].quantile(0.75)
        IQR = Q3 - Q1


        lower_bound = Q1 - outlier_factor * IQR
        upper_bound = Q3 + outlier_factor * IQR

        if keep_outliers:
            data_copy = data_copy[(data_copy[col] < lower_bound) | (data_copy[col] > upper_bound)]
        else:
            data_copy = data_copy[(data_copy[col] >= lower_bound) & (data_copy[col] <= upper_bound)]

    return data_copy


data[['BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = data.groupby('Outcome')[['BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].transform(lambda x: x.replace(0, x.mean()))
data = Manual(data)
data = data.reset_index(drop=True)


from sklearn.preprocessing import StandardScaler
sc_female = StandardScaler()
sc_male = StandardScaler()


X_Data = data.drop('Outcome', axis=1).values
sc_female.fit(X_Data)
X_Data = data.drop(['Outcome','Pregnancies'],axis=1).values
sc_male.fit(X_Data)



male_diabetes_model = pickle.load(open(r'.\backend\\male_model.sav','rb'))
female_diabetes_model = pickle.load(open(r'.\backend\\female_model.sav','rb'))


def Validate(validation_list,Gender):
    if Gender == 'female':
        Pregnancy,Glucose,BP,Skin,Insulin,BMI,DPF,Age = validation_list
    else:
        Glucose,BP,Skin,Insulin,BMI,DPF,Age = validation_list
    if Gender == 'female':
        if Pregnancy < 0:
            return 'Invalid Input for Pregnancies'
    if Glucose <= 0:
        return 'Invalid Input for Glucose'
    if BP <= 0:
        return 'Invalid Input for Blood Pressure'
    if Skin <= 0:
        return 'Invalid Input for Blood Pressure'
    if Insulin <= 0:
        return 'Invalid Input for Insulin'
    if BMI <= 0:
        return 'Invalid Input for BMI'
    if DPF <=0:
        return 'Invalid Input for DPF'
    if Age <=0:
        return 'Invalid Input for Age'
    return 'true'
def Empty_Validation(data,Gender):
    if Gender == 'female':
        Pregnancy = data.get('feature1')
    Glucose = data.get('feature2')
    BP = data.get('feature3')
    Skin = data.get('feature4')
    Insulin = data.get('feature5')
    BMI = data.get('feature6')
    Age = data.get('feature8')
    DPF = data.get('feature7')
    if Gender == 'female':
        if Pregnancy == '' or Glucose == '' or BP == '' or Skin == '' or Insulin == '' or BMI == '' or DPF == '' or Age == '':
            return 'Empty Inputs'
    else:
        if Glucose == '' or BP == '' or Skin == '' or Insulin == '' or BMI == '' or DPF == '' or Age == '':
            return 'Empty Inputs'
    return 'true'

def isGoodDataType(input):
    dot_count = 0
    neg_count = 0
    for i in input:
        if i>='0' and i<='9':
            continue
        if i=='.':
            dot_count += 1
            if dot_count ==1:
                continue
        if i=='-':
            neg_count += 1
            if neg_count ==1:
                continue
        return False
    return True

def Data_type_Validation(data,Gender):
    inputs = []
    if Gender == 'female':
        Pregnancy = data.get('feature1')
        inputs.append(Pregnancy)        
    Glucose = data.get('feature2')
    BP = data.get('feature3')
    Skin = data.get('feature4')
    Insulin = data.get('feature5')
    BMI = data.get('feature6')
    Age = data.get('feature8')
    DPF = data.get('feature7')
    inputs.append(Glucose)
    inputs.append(BP)
    inputs.append(Skin)
    inputs.append(Insulin)
    inputs.append(BMI)
    inputs.append(Age)
    inputs.append(DPF)

    for i in inputs:
        if isGoodDataType(i)== False:
            return 'false'
    return 'true'   
    

@app.route('/')
def home():
    return jsonify({"Ref":"Hello World"})

@app.route('/predict/female', methods=['POST'])
def predict_female():
    if request.method == 'POST':

        data = request.json

        validation_result = Empty_Validation(data,'female')
        if validation_result != 'true':
            return jsonify({'error':validation_result})
        validation_result = Data_type_Validation(data,'female')
        if validation_result != 'true':
            return jsonify({'error':'Invalid Data Type Input'})
        Pregnancy = float(data.get('feature1'))
        Glucose = float(data.get('feature2')) 
        BP =  float(data.get('feature3'))
        Skin = float(data.get('feature4'))
        Insulin = float(data.get('feature5'))
        BMI = float(data.get('feature6'))
        Age = float(data.get('feature8'))
        DPF = float(data.get('feature7'))
        validation_list=[Pregnancy,Glucose,BP,Skin,Insulin,BMI,DPF,Age]
        validation_result = Validate(validation_list,'female')
        if validation_result != 'true':
            return jsonify({'error':validation_result})
        
        prediction = female_diabetes_model.predict(sc_female.transform([[Pregnancy,Glucose,BP,Skin,Insulin,BMI,DPF,Age]]))
        prediction_new = female_diabetes_model.predict_proba(sc_female.transform([[Pregnancy,Glucose,BP,Skin,Insulin,BMI,DPF,Age]]))
        print(prediction)
        if prediction[0] == 1:
            result = {"prediction": "The patient is at the risk of having diabetes",
                      "probability": round(prediction_new[0][1] * 100, 2)}
            return jsonify(result)
        result = {"prediction": "The patient is healthy","probability":"0"}
        return jsonify(result)

@app.route('/predict/male', methods=['POST'])
def predict_male():
    if request.method == 'POST':
        data = request.json

        validation_result = Empty_Validation(data,'male')
        if validation_result != 'true':
            return jsonify({'error':validation_result})
        validation_result = Data_type_Validation(data,'male')
        if validation_result != 'true':
            return jsonify({'error':'Invalid Data Type Input'})
        Glucose = float(data.get('feature2')) 
        BP =  float(data.get('feature3'))
        Skin = float(data.get('feature4'))
        Insulin = float(data.get('feature5'))
        BMI = float(data.get('feature6'))
        Age = float(data.get('feature8'))
        DPF = float(data.get('feature7'))
        validation_list=[Glucose,BP,Skin,Insulin,BMI,DPF,Age]
        validation_result = Validate(validation_list,'male')
        if validation_result != 'true':
            return jsonify({'error':validation_result})
        prediction = male_diabetes_model.predict(sc_male.transform([[Glucose,BP,Skin,Insulin,BMI,DPF,Age]]))
        prediction_new = male_diabetes_model.predict_proba(sc_male.transform([[Glucose,BP,Skin,Insulin,BMI,DPF,Age]]))
        print(prediction)
        if prediction[0] == 1:
            result = {"prediction": "The patient is at the risk of having diabetes",
                      "probability": round(prediction_new[0][1] * 100, 2)}
            return jsonify(result)
        result = {"prediction": "The patient is healthy","probability":"0"}
        return jsonify(result)
    
if __name__ == '__main__':
    app.run(debug=True)
