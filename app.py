import os
import sqlite3

import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, render_template, jsonify, request, Blueprint, session
from keras.models import load_model

from user import user_endpoints

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    gpu0 = gpus[0]  # 如果有多个GPU，仅使用第0个GPU
    tf.config.experimental.set_memory_growth(gpu0, True)  # 设置GPU显存用量按需使用
    tf.config.set_visible_devices([gpu0], "GPU")

print('gpus:', gpus)

img_size = 224

# 标签类比
class_name_dict = ['cataract', 'diabetes', 'glaucoma', 'hypertension', 'myopia', 'normal']

# 标签类比
disease_labels = {
    0: '白内障（Cataract）',
    1: '糖尿病视网膜病变（Diabetic Retinopathy）',
    2: '青光眼（Glaucoma）',
    3: '高血压（Hypertension）',
    4: '近视（Myopia）',
    5: '正常（Normal）'
}

app = Flask(__name__)
login_name = None

# 初始化认证相关路由
user_endpoints(app)


def app_blueprint():
    blueprint = Blueprint('predict', __name__)

    # 加载训练好的模型权重
    print("load model weights...")
    model = load_model('best_model.h5')

    def sigmoid(x):
        s = 1 / (1 + np.exp(-x))
        return s

    @blueprint.route('/submit_and_predict', methods=['POST'])
    def submit_and_predict():
        """
        图像识别预测
        """
        # 获取上传的文件
        test_file = request.files['file']
        filename = test_file.filename

        # 保存上传的文件到指定路径
        save_dir = './static/img/predict_test/'  # 指定保存路径
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)  # 如果目录不存在，创建目录

        test_file_path = os.path.join(save_dir, filename)
        test_file.save(test_file_path)

        img = Image.open(test_file_path)
        img = np.array(img)

        # 模型推理
        image = tf.image.resize(img, [img_size, img_size])
        test_X = tf.expand_dims(image, 0)

        predictions = model.predict(test_X)[0]
        predictions = sigmoid(predictions)
        total = sum(predictions)

        pred_pros = [x / total for x in predictions]
        pred_label = pred_pros.index(max(pred_pros))
        pred_class = disease_labels[pred_label]
        print("预测结果为：", pred_class)
        print(test_file_path)

        user_name = session.get('login_name')  # 从session获取当前用户

        # 通过查询路径检查图像是否已存在本地
        conn = sqlite3.connect('user_info.db')
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM user_uploads WHERE image_path = ?", (test_file_path,))
        result = cursor.fetchone()
        if result[0] > 0:
            conn.close()
            return jsonify({"error": "本地文件夹中已存在相同的图像文件"}), 400

        # 将图像记录插入数据库
        cursor.execute(
            "INSERT INTO user_uploads (image_name, image_path, disease_class, user_name) VALUES (?, ?, ?, ?)",
            (filename, test_file_path, pred_class, user_name)
        )
        conn.commit()
        conn.close()

        result = {
            "upload_image": test_file_path,
            "predict": pred_class,
            "pred_pros": pred_pros,
            "disease_labels": [d.split('（')[0] for d in disease_labels.values()]
        }
        return jsonify(result)

    @blueprint.route('/')
    def index():
        return render_template('index.html')

    @blueprint.route('/test_predict')
    def test_predict():
        return render_template('test_predict.html')

    return blueprint
