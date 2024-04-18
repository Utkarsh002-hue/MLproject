from flask import Flask, request, render_template, jsonify
# Alternatively can use Django, FastAPI, or anything similar
from src.pipeline.pred_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def home_page():
    return render_template('index.html')
@app.route('/predict', methods = ['POST', "GET"])

def predict_datapoint(): 
    if request.method == "GET": 
        return render_template("forms.html")
    else: 
        data = CustomData(
            stars = float(request.form.get('stars')),
            reviews = float(request.form.get('reviews')),
            boughtInLastMonth = float(request.form.get('boughtInLastMonth')),
            category = request.form.get("category"), 
            isBestSeller = request.form.get("isBestSeller"),
        )
    new_data = data.get_data_as_dataframe()
    predict_pipeline = PredictPipeline()
    pred = predict_pipeline.predict_xyz(new_data)


    results = round(pred[0],2)
    

    return render_template("results.html", final_result = results)

if __name__ == "__main__": 
    app.run(host = "0.0.0.0", debug= True)

#http://127.0.0.1:5000/ in browser