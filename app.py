from flask  import Flask, request,jsonify
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


# import pickle
import pickle

#read modle in read-binary mode
model = joblib.load(open('random_model.pkl','rb'))
feature_extraction = joblib.load('vectorizer.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello World"

@app.route('/predict',methods = ['POST'])
def predict():
    message = request.form.get('message')

    input_query = np.array(list([message]))
    input_data_features = feature_extraction.transform(input_query)

    result = model.predict(input_data_features)[0]

    return jsonify({'type':str(result)})

if __name__ == '__main__':
    app.run(debug = True)