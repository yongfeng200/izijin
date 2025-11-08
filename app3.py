import os
import torch
from flask import Flask, render_template, jsonify, request, Blueprint
from torchvision import transforms
from PIL import Image
from UNet import AttentionUNet

# GPU configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

# 图像预处理配置
img_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

app = Flask(__name__)


# 加载UNet模型
def load_unet_model(model_path='UNet_model.pth'):
    model = AttentionUNet(n_channels=1, n_classes=1)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()
    return model


model = load_unet_model()


def app3_blueprint():
    blueprint = Blueprint('segment', __name__)

    @blueprint.route('/submit_and_segment', methods=['POST'])
    def submit_and_segment():
        # 获取上传的文件
        test_file = request.files['file']
        filename = test_file.filename

        # 保存上传的文件到指定路径
        save_dir = './static/img/segment_test/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        test_file_path = os.path.join(save_dir, filename)

        # 检查文件是否已经存储在本地
        if os.path.exists(test_file_path):
            return jsonify({"error": "本地文件夹中已存在相同的图像文件"}), 400

        # 保存文件到本地
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        test_file.save(test_file_path)

        # 图像预处理
        image = Image.open(test_file_path).convert("L")  # 转换为灰度图
        image = image.resize((224, 224))  # 调整图像大小
        image_tensor = img_transform(image).unsqueeze(0).to(device)

        # 模型推理
        with torch.no_grad():
            output = model(image_tensor)
            mask = (output > 0.5).float().squeeze().cpu().numpy()

        # 保存分割图像
        mask_img = Image.fromarray((mask * 255).astype('uint8'), mode='L')
        mask_path = os.path.join(save_dir, 'mask_' + filename)
        mask_img.save(mask_path)

        result = {
            "original_image": test_file_path,
            "mask_image": mask_path
        }
        return jsonify(result)

    @blueprint.route('/test_segment')
    def test_segment():
        return render_template('test_segment.html')

    return blueprint
