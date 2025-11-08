import base64
import random
import sqlite3
import string
from datetime import datetime

import pytz
from captcha.image import ImageCaptcha
from flask import jsonify, redirect, render_template, session, request


def random_char(char_num=4):
    characters = string.ascii_letters + string.digits  # 包含大小写字母和数字
    return ''.join(random.choice(characters) for _ in range(char_num))


# 全局变量，用于存储验证码和登录状态
captcha_text = None
login_name = None


def user_endpoints(app):
    @app.route('/generate_captcha')
    def generate_captcha():
        """生成验证码"""
        global captcha_text
        code = random_char(4)
        captcha_text = code  # 存储验证码文字

        # 创建验证码图片
        image = ImageCaptcha(width=160, height=60)
        captcha_image = image.generate(captcha_text)
        image_bytes = captcha_image.getvalue()

        # 将图片数据编码为 Base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        # 返回包含验证码和图片的 JSON
        return jsonify({
            "code": captcha_text,
            "image": base64_image
        })

    @app.route('/check_captcha/<user_code>')
    def check_captcha(user_code):
        """检查验证码是否正确"""
        global captcha_text
        if user_code == captcha_text.lower():
            return {'status': 'ok'}
        else:
            return {'status': 'error', 'message': '验证码错误'}

    @app.route('/check_login')
    def check_login():
        """判断用户是否登录"""
        return jsonify({
            'username': session.get('login_name'),
            'login': 'login_name' in session
        })

    @app.route('/logout')
    def logout():
        session.pop('login_name', None)  # 从session中移除用户信息
        return redirect('/')  # 重定向到首页

    @app.route('/register')
    def reg():
        return render_template('register.html')

    @app.route('/register/<user_name>/<password>/<verify>/<phone>/<name>')
    def register(user_name, password, verify, phone, name):
        # 检查验证码
        response = check_captcha(verify)
        if response['status'] == 'error':
            return jsonify({'info': '验证码错误', 'status': 'error'})

        conn = sqlite3.connect('user_info.db')
        cursor = conn.cursor()
        check_sql = "SELECT * FROM sqlite_master where type='table' and name='user'"
        cursor.execute(check_sql)
        results = cursor.fetchall()
        # 数据库表不存在
        if len(results) == 0:
            # 创建数据库表
            sql = """
                    CREATE TABLE user(
                        user_name CHAR(256),
                        password CHAR(256),
                        phone CHAR(20),
                        name CHAR(256)
                    );
                    """
            cursor.execute(sql)
            conn.commit()

        sql = "INSERT INTO user (user_name, password, phone, name) VALUES (?,?,?,?);"
        cursor.executemany(sql, [(user_name, password, phone, name)])
        conn.commit()
        conn.close()  # 关闭数据库连接
        return jsonify({'info': '用户注册成功！', 'status': 'ok'})

    @app.route('/login/<user_name>/<password>/<verify>')
    def login(user_name, password, verify):
        # 检查验证码
        response = check_captcha(verify)
        if response['status'] == 'error':
            return jsonify({'info': '验证码错误', 'status': 'error'})

        global login_name
        conn = sqlite3.connect('user_info.db')
        cursor = conn.cursor()
        check_sql = "SELECT * FROM sqlite_master WHERE type='table' AND name='user'"
        cursor.execute(check_sql)
        results = cursor.fetchall()
        if len(results) == 0:
            # 如果没有用户表，则创建
            sql = """
                    CREATE TABLE user(
                        user_name CHAR(256),
                        password CHAR(256),
                        phone CHAR(20),
                        name CHAR(256)
                );
              """
            cursor.execute(sql)
            conn.commit()

        cursor.execute("SELECT * FROM user WHERE user_name=? AND password=?", (user_name, password))
        results = cursor.fetchall()
        conn.close()  # 关闭数据库连接
        if results:
            session['login_name'] = user_name
            return jsonify({'info': user_name + '用户登录成功！', 'status': 'ok'})
        else:
            return jsonify({'info': '用户名或密码错误', 'status': 'error'})

    @app.route('/get_uploads')
    def get_uploads():
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))
        disease_class = request.args.get('disease_class', '')
        user_name = request.args.get('user_name', '')

        conn = sqlite3.connect('user_info.db')
        cursor = conn.cursor()

        # 创建表（如果不存在）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_uploads(
                image_name CHAR(256) NOT NULL,
                image_path CHAR(256),
                disease_class CHAR(256),
                user_name CHAR(256) NOT NULL,
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

        # 构造查询语句
        query = "SELECT image_name, image_path, disease_class, user_name, upload_time FROM user_uploads WHERE 1=1"
        params = []
        # 根据疾病类别查询
        if disease_class:
            query += " AND disease_class LIKE ?"
            params.append(f"%{disease_class}%")
        # 根据上传的用户查询
        if user_name:
            query += " AND user_name LIKE ?"
            params.append(f"%{user_name}%")
        query += " ORDER BY upload_time DESC"

        # 获取总记录数量
        count_query = "SELECT COUNT(*) FROM user_uploads WHERE 1=1"
        if disease_class:
            count_query += " AND disease_class LIKE ?"
        if user_name:
            count_query += " AND user_name LIKE ?"
        cursor.execute(count_query, params)

        total_items = cursor.fetchone()[0]
        totalPages = (total_items + per_page - 1) // per_page

        # 获取分页数据
        query += " LIMIT ? OFFSET ?"
        params.extend([per_page, (page - 1) * per_page])
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()

        uploads = []
        for row in results:
            # 将上传时间转换为北京时间
            time_str = row[4]
            time_utc = datetime.fromisoformat(time_str).replace(tzinfo=pytz.utc)
            time_bj = time_utc.astimezone(pytz.timezone('Asia/Shanghai'))
            # 格式化为标准时间字符串
            time_display = time_bj.strftime("%Y-%m-%d %H:%M:%S")

            uploads.append({
                'image_name': row[0],
                'image_path': row[1],
                'disease_class': row[2],
                'user_name': row[3],
                'upload_time': time_display
            })

        return jsonify({
            'uploads': uploads,
            'totalPages': totalPages
        })