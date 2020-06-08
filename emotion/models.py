from django.db import models

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from PIL import Image
import io, base64

#coral####################################
from PIL import Image
import tflite_runtime.interpreter as tflite
import platform

#環境に合わせてモジュールを読み込む
#from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import load_img, img_to_array
###########################################

#graph = tf.get_default_graph()
graph = tf.compat.v1.get_default_graph()

class Photo(models.Model):
    image = models.ImageField(upload_to='photos')
    IMAGE_SIZE = 100 # 画像サイズ
    MODEL_FILE_PATH = './emotion/ml_models/converted_model.tflite' # モデルファイル
    classes = ["ang", "hap","neu","sad","sur","tir"]
    num_classes = len(classes)

    # 引数から画像ファイルを参照して読み込む
    def predict(self):
        model = None
        global graph
        with graph.as_default():
        
            #edgetpu_runtimeへの紐づけ
            EDGETPU_SHARED_LIB = {'Linux': 'libedgetpu.so.1','Darwin': 'libedgetpu.1.dylib','Windows': 'edgetpu.dll'}[platform.system()]
        
            #モデルの読み込み
            model_file = self.MODEL_FILE_PATH
            model_file, *device = model_file.split('@')

            #tensorflowlitelibAPIへの紐づけ
            interpreter = tflite.Interpreter(model_path=model_file,experimental_delegates=[tflite.load_delegate(EDGETPU_SHARED_LIB,{'device': device[0]} if device else {})])

            #モデルをもとにテンソル展開
            interpreter.allocate_tensors()

            #モデルをもとに入力層・出力層のプロパティを取得
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            input_shape = input_details[0]['shape']

            #photoformから画像データを参照・変換
            img_data = self.image.read()
            img_bin = io.BytesIO(img_data)

            #モデルに合わせた画像データの整形
            image = Image.open(img_bin)
            image = image.convert("RGB")
            image = image.resize((self.IMAGE_SIZE, self.IMAGE_SIZE))
            data = np.asarray(image) / 255.0
            data = np.array(data)

            #tensorflowlitelibAPIの画像読み込み形式に合わせる
            img_1_224_224 = data
            img_1_224_224_3 = img_1_224_224[None, ...] 
            input_data = np.array(img_1_224_224_3,dtype=np.float32)
            
            #推論を行いたい画像のポインタをセット
            interpreter.set_tensor(input_details[0]['index'], input_data)

            #推論実行
            interpreter.invoke()

            #推論結果の取得
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            #推論結果の整形
            predicted = output_data.argmax()
            percentage = str(np.round(output_data[0][predicted]*100,2))
            #predicted1 = output_data.[0]
            #predicted2 = output_data.[1]
            #predicted3 = output_data.[2]
            #predicted4 = output_data.[3]
            #predicted5 = output_data.[4]

            return self.classes[predicted], percentage

    def image_src(self):
        with self.image.open() as img:
            base64_img = base64.b64encode(img.read()).decode()

            return 'data:' + img.file.content_type + ';base64,' + base64_img


