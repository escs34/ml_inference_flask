# -*- coding: UTF-8 -*-
from flask import Flask, render_template, request
#from werkzeug import secure_filename
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/home/')
def home():
    return 'Hello, World!'

@app.route('/upload', methods = ['GET','POST'])
def upload_file():
    if request.method == 'POST':
        file_str = request.files['file'].read()

        npimg = np.fromstring(file_str, np.uint8)

        if np.size(npimg)<100:
            return "Img size is under 100"

        npimg = npimg[:100].reshape(1,-1)
        print(npimg.shape)
        result = my_model.predict(npimg, verbose=0)
        return 'uploads 디렉토리 -> 파일 업로드 성공 and result:  ' + str(result)





if __name__ == '__main__':
    from model import get_model
    global my_model
    my_model=get_model()

    app.run(debug=True,
            host="0.0.0.0",
            port=5000
            )
