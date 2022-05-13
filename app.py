import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn import preprocessing

app = Flask(__name__)
CORS(app)
from model_script import SARIMAX_model

@app.route("/")
def hello():
  return "Hello World!"


#@app.route('/home', methods=['GET'])
#def home():
 #    data = request.args.get(data)
  #   
   #  
    # 
     #return jsonify(data)
@app.route("/get_prediction", methods=['POST','OPTIONS'])
def get_prediction():

    df = pd.DataFrame(request.json, index=[0])

    cols=["drugName","total_weekly_stock", "week"]
    drugName = (df['drugName'])
    week = (df['week'])
    total_weekly_stock = (df['week'])
    
    label_encoder = preprocessing.LabelEncoder()
    return jsonify({'result': SARIMAX_model.predict(df)[0]}), 201


 #    json_ = request.json
  #   query_df = pd.DataFrame(json_)
   #  query = pd.get_dummies(query_df)

     
    # return render_template('index.html', prediction_text=data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)
