"""
Flask Serving
This file is a sample flask app that can be used to test your model with an API.
This app does the following:
    - Handles uploads and looks for an image file send as "file" parameter
    - Stores the image at ./images dir
    - Invokes super_resolve function from super_resolve.py with this image
    - Returns the output file generated at /output
Additional configuration:
    - You can also choose the model file name to use as a request parameter
    - Parameter name: model
    - It is loaded from /models
"""
import os
from flask import Flask, send_file, request
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename

from super_resolve import super_resolve

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])
app = Flask(__name__)


@app.route('/<path:path>', methods=["POST"])
def super_resolve_service(path):
    """
    Take the input image and super resolve it
    """
    # check if the post request has the file part
    if 'file' not in request.files:
        return BadRequest("File not present in request")
    file = request.files['file']
    if file.filename == '':
        return BadRequest("File name is not present in request")
    if not allowed_file(file.filename):
        return BadRequest("Invalid file type")
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        input_filepath = os.path.join('./', filename)
        output_filepath = os.path.join('/output/', filename)
        file.save(input_filepath)

        # Get checkpoint filename from la_muse
        model = request.form.get("model") or "model_epoch_10.pth"
        super_resolve(input_filepath, output_filepath, '/models/' + model, True)
        return send_file(output_filepath, mimetype='image/jpg')


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(host='0.0.0.0')