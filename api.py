from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)


# =============================================================================
# Steps for building the Flask API for cricket match outcome prediction
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