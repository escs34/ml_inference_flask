# -*- coding: UTF-8 -*-
from flask import Flask, render_template, request
import numpy as np
import time

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

        #original
        npimg = np.fromstring(file_str, np.float32)

        if np.size(npimg)<32:
            return "Img size is under 32, " + str(file_str)

        npimg = npimg.reshape(1,32,32,3)
        print(npimg.shape)
        result = my_model.predict(npimg, verbose=0)
        return 'uploads 디렉토리 -> 파일 업로드 성공 and result:  ' + str(result) + "\n\n"


@app.route('/upload_jpg', methods = ['GET','POST'])
def upload_file_jpg():
    if request.method == 'POST':
        
        process_start = time.time()
        file_str = request.files['file'].read()
        
        #get from multipart means getting bytes of image.
        #it must be decoded to normal image.
        #this task can be done by opencv or PIL.Image
        #this function use PIL.Image.
        
        data = np.fromstring(file_str, dtype=np.uint8)
        
        import io
        data_io = io.BytesIO(data)
        
        from PIL import Image
        img = Image.open(data_io)
        npimg = np.array(img)
        
        
        #original
        if np.size(npimg)<32:
            return "Img size is under 32, " + str(file_str)

        npimg = npimg.reshape(1,32,32,3)
        print(npimg.shape)

        if np.mean(average_time) > 5:
            print("model : 0")
            result = my_models[-1].predict(npimg, verbose=0)
        else:
            print("model : last")
            result = my_models[0].predict(npimg, verbose=0)
            
        process_end = time.time()

        average_time.append(process_end - process_start)

        if len(average_time) >50 :
            average_time.clear()
        
        
        return 'uploads jpg 디렉토리 -> 파일 업로드 성공 and result:  ' + str(result) + "\n\n"




if __name__ == '__main__':
    import tensorflow.keras.models as models
    import tensorflow as tf
    global my_models
    
    my_models = []

    with tf.device('/cpu:0'):
        total_model= models.load_model("ResNet164_v1_spinal.h5")

        for each_output in total_model.outputs:
            tmp_model = tf.keras.models.Model(inputs=total_model.input, outputs=each_output)
            my_models.append(tmp_model)

    global average_time
    average_time = []

    app.run(debug=True,
            host="0.0.0.0",
            port=5000
            )
