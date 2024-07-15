import numpy as np
import pickle
import pandas as pd
import os
from flask import Flask,request,render_template


app=Flask(__name__)
model = pickle.load(open('honmodel.pkl','rb'))


@app.route('/')
def home():
     return render_template('index.html')


@app.route('/inner_page',methods=["POST","GET"])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_values = [np.array(input_features) ]
    print(features_values)
    
    names=['cs','D','w','p','E','f','g','po','Vis','pu']
              
   
    df = pd.DataFrame(features_values, columns= names)
    
    prediction = model.predict(df)
    print(prediction[0])
    rounded_value = round(prediction[0], 2)
    text="Hence,based on calculation, the predicted Honey Price is: "
    
    return render_template('inner_page.html', prediction_text=text+str(rounded_value))



if __name__ == '__main__':
    app.run(debug = False, port=5000)

  


 