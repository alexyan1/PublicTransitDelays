from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from datetime import datetime

app = Flask(__name__)

# Load the trained pipeline
pipeline = joblib.load('transport_delay_pipeline.pkl')

def create_input(stop_name, line_name, datetime_str, distance=0.5):
    """Create input DataFrame matching training features"""
    dt = datetime.strptime(datetime_str, '%Y-%m-%dT%H:%M')
    return pd.DataFrame([{
        'PublishedLineName': line_name,
        'NextStopPointName': stop_name,
        'DayOfWeek': dt.weekday(),
        'Hour': dt.hour,
        'DistanceFromStop': distance
    }])

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Get form data
            stop = request.form['stop']
            line = request.form['line']
            dt = request.form['datetime']
            
            # Create input DataFrame
            input_df = create_input(stop, line, dt)
            
            # Make prediction
            prediction = pipeline.predict(input_df)[0]
            return render_template('result.html', 
                                 prediction=round(prediction, 1),
                                 stop=stop,
                                 line=line,
                                 time=dt)
        except Exception as e:
            return render_template('error.html', error=str(e))
    
    # Show form for GET requests
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict_api():
    """API endpoint for JSON requests"""
    try:
        data = request.json
        input_df = create_input(
            data['stop'],
            data['line'],
            data['datetime'],
            data.get('distance', 0.5)
        )
        prediction = pipeline.predict(input_df)[0]
        return jsonify({
            'stop': data['stop'],
            'line': data['line'],
            'predicted_delay': round(prediction, 1),
            'units': 'minutes'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)