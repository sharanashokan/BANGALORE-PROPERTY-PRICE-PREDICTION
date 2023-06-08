from flask import Flask, render_template, request
import pickle
import json
import numpy as np

app = Flask(__name__)

# Load the pre-trained model during application startup
model = pickle.load(open('bhpkl.pickle', 'rb'))

# Load the column names from the JSON file
with open('columns.json', 'r') as file:
    columns_data = json.load(file)
    data_columns = columns_data['data_columns']

def predict_price(location, sqft, bath, bhk):
    # Define and initialize x variable
    x = np.zeros(len(data_columns))

    # Find index of location in data_columns
    loc_index = data_columns.index(location)

    # Assign input values to x
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    x[loc_index + 3] = 1  # Set the value of the corresponding location index to 1

    # Predict the price using the loaded model
    price = model.predict([x])[0]
    rounded_price = round(price, 2)  # Round the price to 2 decimal places
    formatted_price = "{} Lakhs".format(rounded_price)  # Append "Lakhs" to the rounded price

    return formatted_price

@app.route('/')
def home():
    return render_template('index.html', location_names=data_columns[3:])

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the HTML form
    location = request.form['location']
    total_sqft = float(request.form['total_sqft'])
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])
    
    # Make predictions using the modified predict_price function
    price = predict_price(location, total_sqft, bath, bhk)
    
    # Render the result page with the predicted price
    return render_template('index.html', price=price, location_names=data_columns[3:])

if __name__ == '__main__':
    app.run(debug=True)
