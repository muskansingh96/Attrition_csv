from flask import Flask, render_template, request
import pandas as pd
import pickle 
from sklearn.preprocessing import LabelEncoder
import csv
model = pickle.load(open('rforest_model.pkl', 'rb'))
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/data',methods=['GET', 'POST'])
def data():
    if request.method == 'POST':
        f = request.form['csvfile']
        data = []
        with open(f) as file:
            csvfile = csv.reader(file)
            for row in csvfile:
                data.append(row)
                
        data = pd.DataFrame(data)
        
        
        #df.rename(columns=df.iloc[0]).drop(df.index[0])
        headers = data.iloc[0]
        print(headers)
        data  = pd.DataFrame(data.values[1:], columns=headers)
        
        
        #renaming column name 
        data.rename(columns={'sales': 'department'}, inplace=True)
        data = data.loc[:, data.columns != '']
        
        print(type(data))
        print(data.columns)
        print(data.dtypes)
        print(data.head(5))
        Analysis_Df = data
        
        # Label encoding
        # salary = {'low':0, 'medium':1,'high':2}
        # labelencoder = LabelEncoder()
        # data['salary'] = data['salary'].map(lambda x : salary[x])
        
        data.loc[data['salary']=='low' , 'salary'] = 0
        data.loc[data.salary=='medium' , 'salary'] = 1
        data.loc[data.salary=='high' , 'salary'] = 2
        
        data['salary'] = data.salary.astype(float)
        data['satisfaction_level'] =data.satisfaction_level.astype(float)
        data['last_evaluation'] = data.last_evaluation.astype(float)
        data['number_project'] = data.number_project.astype(float)
        data['average_montly_hours'] =data.average_montly_hours.astype(float)
        data['time_spend_company'] = data.time_spend_company.astype(float)
        data['Work_accident'] =data.Work_accident.astype(float)
        data['promotion_last_5years'] = data.promotion_last_5years.astype(float)
        
        
        
        print(data.dtypes)
        #weather["Temp"] = weather.Temp.astype(float)
        
        data=pd.get_dummies(data)
        
        emp_status = model.predict(data)
        
        Analysis_Df['left'] =   pd.Series(emp_status)
        Analysis_Df['  Prediction  '] = "pending" 
        Analysis_Df.loc[Analysis_Df.left==0, '  Prediction  '] = "  staying  "
        Analysis_Df.loc[Analysis_Df.left==1 , '  Prediction  '] = "  leaving  "
        
        print(Analysis_Df.columns)
        
        df_bydept = Analysis_Df.groupby("department")['  Prediction  '].value_counts()
        final_df = df_bydept.to_frame(name=None)
        
        # Adding status col  in dataframe
        #final_df['result'] = "pending"
        
        print()
    
        #final_df.loc[final_df.left==0, 'result'] = "staying"
        #final_df.loc[final_df.left==1 , 'result'] = "leaving"
        
        #print(final_df['left'])
        print('**********',final_df.columns)
        final_df.rename(columns={'  Prediction  ': 'employee_count'}, inplace=True)
        #data = final_df.loc[:, data.columns != 'status']
        data = final_df
        
        return render_template('data.html', data=data.to_html())
    
if __name__ == '__main__':
    app.run(debug=True)
                
        