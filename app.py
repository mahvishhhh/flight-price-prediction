
from flask import Flask, request, render_template
from flask_cors import cross_origin
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("/home/fathima/projects/flight-price-prediction/artifacts/pred_model.pkl", "rb"))

@app.route("/")
@cross_origin()
def home():
    return render_template("index.html")

def encode_categorical_values(input_dict, value_map):
    encoded_values = {k: 0 for k in value_map.values()}
    if input_dict in value_map:
        encoded_values[value_map[input_dict]] = 1
    else:
        raise Exception(f"Value '{input_dict}' not recognized")
    return encoded_values

def calculate_duration_in_minutes(dep_time, arr_time):
    dep_datetime = pd.to_datetime(dep_time, format="%Y-%m-%dT%H:%M")
    arr_datetime = pd.to_datetime(arr_time, format="%Y-%m-%dT%H:%M")
    
    duration_minutes = (arr_datetime - dep_datetime).total_seconds() / 60
    return duration_minutes

def process_user_input(request_form):
    dep_time = request_form['Dep_Time']
    dep_datetime = pd.to_datetime(dep_time, format="%Y-%m-%dT%H:%M")

    dep_day = dep_datetime.day
    dep_month = dep_datetime.month
    dep_hour = dep_datetime.hour
    dep_min = dep_datetime.minute

    arr_time = request_form["Arr_Time"]
    arr_datetime = pd.to_datetime(arr_time, format="%Y-%m-%dT%H:%M")

    arr_hour = arr_datetime.hour
    arr_min = arr_datetime.minute

    dur = calculate_duration_in_minutes(dep_time, arr_time)


    Total_stops = int(request_form["stops"])

    Airline = encode_categorical_values(request_form['airline'], {
        'Jet Airways': 'Jet_Airways',
        'IndiGo': 'IndiGo',
        'Air India': 'Air_India',
        'Air Asia': 'Air_Asia',
        'Multiple carriers': 'Multiple_carriers',
        'SpiceJet': 'SpiceJet',
        'Vistara': 'Vistara',
        'GoAir': 'GoAir',
        'Multiple carriers Premium economy': 'Multiple_carriers_Premium_economy',
        'Jet Airways Business': 'Jet_Airways_Business',
        'Vistara Premium economy': 'Vistara_Premium_economy',
        'Trujet': 'Trujet'
    })

    Source = encode_categorical_values(request_form['Source'], {
        'Delhi': 's_Delhi',
        'Kolkata': 's_Kolkata',
        'Mumbai': 's_Mumbai',
        'Chennai': 's_Chennai',
        'Banglore': 's_Banglore'
    })

    Destination = encode_categorical_values(request_form['Destination'], {
        'Cochin': 'd_Cochin',
        'Delhi': 'd_Delhi',
        'New Delhi': 'd_New_Delhi',
        'Hyderabad': 'd_Hyderabad',
        'Kolkata': 'd_Kolkata',
        'Bangalore': 'd_Banglore'
    })

    return [
        Total_stops,
        dep_day,
        dep_month,
        dep_hour,
        dep_min,
        arr_hour,
        arr_min,
        dur,
        *Airline.values(),
        *Source.values(),
        *Destination.values()
    ]

@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    if request.method == "POST":
        features = process_user_input(request.form)

        prediction = model.predict([features])

        output = round(prediction[0], 2)

        return render_template("index.html", prediction_text="Your Flight price is Rs. {}".format(output))

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
