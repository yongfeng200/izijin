import os
import sqlite3

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from flask import Flask, render_template, jsonify, request, Blueprint, session
from torchvision import transforms, models

from user import user_endpoints

# GPU configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

# Image preprocessing configuration
img_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 标签类比
disease_labels = {
    0: '黄斑变性（AMD）',
    1: '脉络膜新生血管（CNV）',
    2: '中央浆液性视网膜病变（CSR）',
    3: '糖尿病黄斑水肿（DME）',
    4: '糖尿病视网膜病变（DR）',
    5: '玻璃膜疣（DRUSEN）',
    6: '黄斑裂孔（MH）',
    7: '健康（HEALTHY）'
}

app = Flask(__name__)
login_name = None

# 初始化认证相关路由
user_endpoints(app)

# 加载分类对应的索引
class_to_idx = torch.load('class_to_idx.pth')
idx_to_class = {v: k for k, v in class_to_idx.items()}

# 加载疾病中的标签
disease_labels = {
    v: disease_labels[idx]  # 使用idx来获取中文标签
    for idx, v in enumerate(class_to_idx.values())
}


def app2_blueprint():
    blueprint = Blueprint('predict2', __name__)

    # 加载模型
    def load_model(model_path='resnet18_model.pth', num_classes=8):
        model2 = models.resnet18(pretrained=False)
        num_ftr = model2.fc.in_features
        model2.fc = nn.Linear(num_ftr, num_classes)
        model2.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model2.to(device)
        model2.eval()
        return model2

    model = load_model()

    # 图像预测
    @blueprint.route('/submit_and_predict', methods=['POST'])
    def submit_and_predict():
        # 获取上传的文件
        test_file = request.files['file']
        filename = test_file.filename

        # 保存上传的文件到指定路径
        save_dir = './static/img/predict_test2/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        test_file_path = os.path.join(save_dir, filename)
        test_file.save(test_file_path)

        # 图像预处理
        image = Image.open(test_file_path).convert('RGB')
        image_tensor = img_transform(image).unsqueeze(0).to(device)

        # 模型推理
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)

            pred_pros = probabilities[0].cpu().numpy().tolist()
            pred_idx = torch.argmax(probabilities, dim=1).item()
            pred_class = disease_labels[pred_idx]
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

    @blueprint.route('/test_predict2')
    def test_predict():
        return render_template('test_predict2.html')

    return blueprint
