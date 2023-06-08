from flask import Flask, request , jsonify
import util

app = Flask(__name__)

@app.route('/get_location_names')
def get_location_names():
    response = jsonify({
        "location" : util.get_location_names()
    })
    response.headers.add("access-control-allow-origin","*")
    return response


if __name__ == "__main__":
    print("starting python flask server")
    app.run()