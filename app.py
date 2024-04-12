# from flask import Flask, request, jsonify
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.feature_extraction.text import CountVectorizer

# app = Flask(__name__)

# # Load the dataset
# dataset = pd.read_csv('/Users/dev_cs/Documents/desireapp/banks.csv')

# # Preprocess the data
# dataset['Years of Experience'] = dataset['Years of Experience'].astype(str)  # Convert to string
# X = dataset['Education'] + ' ' + dataset['Years of Experience'] + ' ' + dataset['Subjects Taught'] + ' ' + dataset['Certificates']
# y = dataset['Rating']

# # Vectorize the resume text
# vectorizer = CountVectorizer()
# X_vectorized = vectorizer.fit_transform(X)

# # Train the model
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_vectorized, y)

# @app.route('/rate_teachers', methods=['POST'])
# def rate_teachers():
#     data = request.get_json()
#     # name = newcv = data.get('name') # Get the JSON payload
    
#     newcv = data.get('newcv')  # Extract the 'newcv' data from the payload
    
#     # Preprocess the example resume
#     resume_vectorized = vectorizer.transform([newcv])

#     # Predict the rating
#     rating_prediction = model.predict(resume_vectorized)
    
#     return jsonify({'rating': rating_prediction[0]})

# if __name__ == '__main__':
#     app.run()
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib  # For model persistence
import requests
# Import LlamaAPI class from llamaapi module
from llamaapi import LlamaAPI
from openai import OpenAI


app = Flask(__name__)



print(response.choices[0].message.content)
# Load the dataset
dataset = pd.read_csv('/Users/dev_cs/Documents/desireapp/banks.csv')

# Preprocess the data
dataset['Years of Experience'] = dataset['Years of Experience'].astype(str)  # Convert to string
X = dataset['Education'] + ' ' + dataset['Years of Experience'] + ' ' + dataset['Subjects Taught'] + ' ' + dataset['Certificates']
y = dataset['Rating']

# Vectorize the resume text
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Persist the model to disk for future use
joblib.dump(model, 'teacher_rating_model.joblib')
joblib.dump(vectorizer, 'resume_vectorizer.joblib')

@app.route('/rate_teachers', methods=['POST'])
def rate_teachers():
    

    # Process request from client
    data = request.get_json()
    if 'newcv' not in data:
        return jsonify({'error': 'Invalid request. Missing "newcv" data.'}), 400
    
    newcv = data['newcv']  # Extract the 'newcv' data from the payload
    
    try:
       
    
       


        resume_vectorized = vectorizer.transform([newcv])
        rating_prediction = model.predict(resume_vectorized)
        
        return jsonify({'rating_from_model': rating_prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)  # Set debug=True for development purposes
