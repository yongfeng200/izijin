import os

from flask import Flask, render_template, Blueprint

from app import app_blueprint
from app2 import app2_blueprint
from app3 import app3_blueprint

from user import user_endpoints

app = Flask(__name__)
app.secret_key = os.urandom(24)

# 初始化认证相关路由
user_endpoints(app)
blueprint = Blueprint('predict', __name__, static_folder='static', static_url_path='/static')

# 注册 app 蓝图
eyes = app_blueprint()
app.register_blueprint(eyes, url_prefix='/predict')

# 注册 app2 蓝图
retina = app2_blueprint()
app.register_blueprint(retina, url_prefix='/predict2')

# 注册 app3 蓝图
segment = app3_blueprint()
app.register_blueprint(segment, url_prefix='/segment')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload')
def upload():
    return render_template('upload.html')


if __name__ == "__main__":
    app.run(host='127.0.0.1', debug=True)
