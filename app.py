from flask import Flask, render_template, request, send_file
from src.exception import CustomException
from src.logger import logging as lg
import os,sys

from src.pipeline.training_pipeline import TrainPipeline
from src.pipeline.predict_pipeline import PredictionPipeline

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/train")
def train_route():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()

        return render_template("training.html")

    except Exception as e:
        raise CustomException(e,sys)

@app.route('/predict', methods=['POST', 'GET'])
def upload():
    
    try:


        if request.method == 'POST':
            # it is a object of prediction pipeline
            prediction_pipeline = PredictionPipeline(request)
            
            #now we are running this run pipeline method
            prediction_file_detail = prediction_pipeline.run_pipeline()

            lg.info("prediction completed. Downloading prediction file.")
            return send_file(prediction_file_detail.prediction_file_path,
                            download_name= prediction_file_detail.prediction_filename,
                            as_attachment= True)


        else:
            return render_template('upload_file.html')
    except Exception as e:
        raise CustomException(e,sys)
    


if __name__ == "__main__":
    app.run(host="0.0.0.0", port = 8080)