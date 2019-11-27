import os
from flask import Flask, render_template, flash, request, redirect, url_for, send_from_directory

from werkzeug.utils import secure_filename
from classifiers import predictions

UPLOAD_FOLDER = os.getcwd() + "\\static\\upload"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = Flask(__name__)
app.upload_folder = UPLOAD_FOLDER
app.secret_key = os.urandom(24)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template("home/index.html")


@app.route('/upload-image', methods=["GET", "POST"])
def upload_image():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'image' not in request.files:
            flash('No image part')
            return redirect(request.url)
        file = request.files['image']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.upload_folder, filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return redirect(url_for('home'))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    path = url_for('static', filename = 'upload/'+filename)
    pred = predictions.predict(filename)
    return render_template("home/index_with_results.html", path = path, predictions = pred)
    # return render_template("home/index.html")


if __name__ =="__main__":
    app.run(debug=True,port=5000)