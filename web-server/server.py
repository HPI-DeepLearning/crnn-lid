# System imports
import sys, subprocess, time
import numpy as np
from os import path
import flask
from flask.ext.cors import CORS
from flask import *
from flask.json import jsonify
from werkzeug import secure_filename
from flask_extensions import *
from keras.models import load_model
import tensorflow as tf

lib_path = os.path.abspath(os.path.join('../keras'))
sys.path.append(lib_path)
from data_loaders.SpectrogramGenerator import SpectrogramGenerator


static_assets_path = path.join(path.dirname(__file__), "dist")
app = Flask(__name__, static_folder= static_assets_path)
CORS(app)


# ----- Routes ----------

@app.route("/", defaults={"fall_through": ""})
@app.route("/<path:fall_through>")
def index(fall_through):
    if fall_through:
        return redirect(url_for("index"))
    else:
        return app.send_static_file("index.html")


@app.route("/dist/<path:asset_path>")
def send_static(asset_path):
    return send_from_directory(static_assets_path, asset_path)


@app.route("/audio/<path:audio_path>")
def send_audio(audio_path):
    return send_file_partial(path.join(app.config["UPLOAD_FOLDER"], audio_path))


@app.route("/api/upload", methods=["POST"])
def uploadAudio():

    def is_allowed(filename):
        return len(filter(lambda ext: ext in filename, ["wav", "mp3", "ogg"])) > 0

    file = request.files.getlist("audio")[0]

    if file and is_allowed(file.filename):
        filename = secure_filename(file.filename)
        file_path = path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # convert_to_mono_wav(file_path, True)

        response = jsonify(get_prediction(file_path))
    else:
        response = bad_request("Invalid file")

    return response


@app.route("/api/example/<int:example_id>")
def use_example(example_id):
    if example_id <= 3:
        filename = "audio%s.wav" % example_id
        file_path = path.join(app.config["UPLOAD_FOLDER"], "examples", filename)
        response = jsonify(get_prediction(file_path))
    else:
        response = bad_request("Invalid Example")

    return response


def bad_request(reason):
    response = jsonify({"error" : reason})
    response.status_code = 400
    return response


# -------- Prediction & Features --------
def get_prediction(file_path):

    LABEL_MAP = {
        0 : "English",
        1 : "German",
        2 : "French",
        3 : "Spanish",
        4 : "Chinese",
        5 : "Russian"

    }

    config = {"pixel_per_second": 50, "input_shape": [129, 500, 1], "num_classes": 4}
    data_generator = SpectrogramGenerator(file_path, config, shuffle=False, run_only_once=True).get_generator()
    data = [np.divide(image, 255.0) for image in data_generator]
    data = np.stack(data)

    # load model and do predictions
    model = getattr(g, "_model", None)
    if model is None:
        model = flask.g._model = load_model("model/2017-01-02-13-39-41.weights.06.model")

    probabilities = flask.g._model.predict(data)

    # average predictions along time axis (majority voting)
    average_prob = np.mean(probabilities, axis=0)
    average_class = np.argmax(average_prob)

    print(probabilities, average_prob, average_class)

    pred_with_label = {LABEL_MAP[index] : prob for index, prob in enumerate(average_prob.tolist())}

    # transform results a little to make them ready for JSON conversion
    file_path = file_path + "?cachebuster=%s" % time.time()
    result = {
        "audio" : {
            "url" : "%s" % file_path,
        },
        "predictions" : pred_with_label
    }

    return result


if __name__ == "__main__":
    app.config.update(
        DEBUG = True,
        SECRET_KEY = "asassdfs",
        CORS_HEADERS = "Content-Type",
        UPLOAD_FOLDER = "audio",
    )
    # Start the server

    # Make sure all frontend assets are compiled
    # subprocess.Popen("webpack")

    # Start the Flask app
    app.run(port=6006)
