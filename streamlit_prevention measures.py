from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load and preprocess data
data = pd.read_csv('C:/Users/pshri/Downloads/Disease precaution.csv')
data['Precautions'] = data[['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].apply(lambda x: ' | '.join(x.dropna()), axis=1)

# Convert 'Disease' column to numerical form using LabelEncoder
le = LabelEncoder()
data['Disease_encoded'] = le.fit_transform(data['Disease'])

# Define features (Disease name) and target (Precautions)
X = data['Disease_encoded'].values.reshape(-1, 1)
y = data['Precautions']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

@app.route('/predict', methods=['POST'])
def predict_precautions():
    # Extract the disease name from the POST request
    data = request.json
    disease_name = data.get('disease_name')
    
    if not disease_name:
        return jsonify({'error': 'Disease name is required'}), 400
    
    if disease_name not in le.classes_:
        return jsonify({'error': 'Disease not found in the dataset'}), 404

    # Encode the disease name to its corresponding encoded value
    disease_encoded = le.transform([disease_name])

    # Predict the precautions
    precautions = rf_model.predict([disease_encoded])
    
    return jsonify({'disease_name': disease_name, 'precautions': precautions[0]})

if __name__ == '__main__':
    app.run(debug=True)
