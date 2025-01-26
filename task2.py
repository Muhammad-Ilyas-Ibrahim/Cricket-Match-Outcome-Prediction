from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)


# =============================================================================
# Step 1: Load the trained model (I saved the model in Task 1 as 'model.pkl')
# Step 2: Initialize the Flask app
# Step 3: Define the API endpoint ('/predict') that accepts POST requests
# Step 4: Load the input CSV file from the request
# Step 5: Filter rows based on the criteria (balls_left < 60 and target > 120)
# Step 6: Run predictions on the filtered rows using the trained model
# Step 7: Save the predictions to a new CSV file
# Step 8: Return the path to the results CSV file as a JSON response
# =============================================================================


model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    
    df = pd.read_csv(file)
    filtered_df = df[(df['balls_left'] < 60) & (df['target'] > 120)]
    predictions = model.predict(filtered_df[['total_runs', 'wickets', 'target', 'balls_left']])
    
    filtered_df['won'] = predictions
    filtered_df.to_csv('results.csv', index=False)
    
    return jsonify({'results_path': 'results.csv'})

if __name__ == '__main__':
    app.run(debug=True)