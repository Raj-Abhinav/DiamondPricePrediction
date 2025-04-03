from flask import Flask, request, render_template, jsonify, send_from_directory
from src.pipelines.prediction_pipeline import CustomData, PredictionPipeline
import os

# Create Flask app with static folder configuration
application = Flask(__name__, 
                   static_folder='static',
                   static_url_path='/static')

app = application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_datapoint():
    try:
        data = CustomData(
            carat=float(request.form.get('carat')),
            depth=float(request.form.get('depth')),
            table=float(request.form.get('table')),
            x=float(request.form.get('x')),
            y=float(request.form.get('y')),
            z=float(request.form.get('z')),
            cut=request.form.get('cut'),
            color=request.form.get('color'),
            clarity=request.form.get('clarity')
        )
        
        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictionPipeline()
        pred = predict_pipeline.predict(final_new_data)
        
        results = round(pred[0], 2)
        return jsonify({'price': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)

