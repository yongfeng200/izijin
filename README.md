# 基于深度学习的眼部OCT影像分析系统

## 项目概述

本项目旨在开发一个基于深度学习的眼部OCT影像分析系统，利用TensorFlow、Keras和PyTorch等深度学习框架构建CNN、ResNet模型进行眼部疾病影像分类，并使用全卷积网络（FCN）进行图像分割，实现眼部疾病的早期诊断与分析。

## 核心功能

  1. **眼部OCT图像分类**
     * 基于CNN和ResNet的深度学习模型，对眼部OCT图像进行分类，识别多种眼部疾病，如AMD、CNV、DME等。
     * 提供模型训练、验证和测试的完整流程，包括数据预处理、模型构建、训练优化和性能评估。

  2. **图像分割**
     * 使用U-Net结合注意力机制，实现眼部OCT图像的像素级分割，精准定位病变区域。
     * 提供分割模型的训练和评估，并支持分割结果的可视化。

  3. **在线分析平台**
     * 基于Flask框架开发的Web应用，支持用户上传OCT图像进行实时分析。
     * 提供用户登录、注册和上传记录查询功能，方便用户管理分析结果。

## 技术栈

  * **深度学习框架** ：TensorFlow 2.9.0、Keras 2.9.0、PyTorch 1.13.1
  * **Web开发** ：Flask 3.1.0
  * **数据库** ：SQLite
  * **前端** ：Bootstrap 5.3.0、jQuery 3.6.0
  * **可视化** ：Matplotlib 3.7.1、Seaborn 0.12.2

## 系统架构

### 后端架构

  1. **Flask应用** ：负责处理HTTP请求，提供API接口和服务端逻辑。
     * **用户认证模块** ：实现用户登录、注册、登出等功能。
     * **图像分析模块** ：接收用户上传的图像，调用深度学习模型进行分析，返回分析结果。
     * **数据管理模块** ：管理用户上传的图像和分析记录，存储在SQLite数据库中。

  2. **深度学习模型** ：
     * **CNN模型** ：基于Keras构建的卷积神经网络，用于眼部OCT图像分类。
     * **ResNet模型** ：使用PyTorch以及Keras实现的残差神经网络，提升分类性能。
     * **U-Net模型** ：基于PyTorch的全卷积网络，用于图像分割任务。

### 前端架构

  1. **HTML/CSS/JS** ：构建响应式用户界面，提供良好的用户体验。
  2. **Bootstrap框架** ：实现页面布局和组件样式，确保在不同设备上的兼容性。
  3. **jQuery** ：处理页面交互和Ajax请求，实现动态数据加载和更新。

## 使用方法

### 环境准备

  1. 创建并激活虚拟环境：

     * `python -m venv myenv`
     * `source myenv/bin/activate`（Linux/Mac）
     * `myenv\Scripts\activate`（Windows）

  2. 安装依赖包：

     * `pip install -r requirements.txt`

### 启动服务

`python main.py`

服务默认运行在`<http://127.0.0.1:5000>`，可以通过浏览器访问。

### 用户操作流程

  1. **注册 / 登录** ：访问`/register`进行用户注册，访问`/`进行登录。
  2. **图像分析** ：访问`/predict/test_predict`进行眼底彩照分析，访问`/predict2/test_predict2`进行OCT图像分析，访问`/segment/test_segment`进行图像分割。
  3. **查看记录** ：访问`/upload`查看上传记录和分析结果。

## 性能评估

### 分类模型性能

| 模型类型 | 平均AUC |
|----------|---------|
| CNN模型 | 0.989 |
| ResNet模型 | 0.999 |

### 分割模型性能

| 指标                | 分割模型 |
|---------------------|----------|
| 平均Dice系数         | 0.8263   |
| 平均IoU              | 0.7085   |

## 项目目录结构

```
eye-disease-classify/
├── app.py              # 眼底图像分类相关路由和逻辑
├── app2.py             # OCT图像分类相关路由和逻辑
├── app3.py             # 图像分割相关路由和逻辑
├── main.py             # 项目入口文件
├── train.py            # 模型训练脚本
├── UNet.py             # U-Net模型定义
├── user.py             # 用户认证相关功能
├── static/             # 静态文件目录
│   ├── css/            # CSS文件
│   ├── js/             # JavaScript文件
│   ├── img/            # 图像文件
│   └── assets/         # 其他静态资源
├── templates/          # HTML模板文件
├── requirements.txt    # 项目依赖包
└── README.md           # 项目说明文档
```

## 联系方式

如需进一步交流或合作，请通过以下方式联系：

  * 邮箱：[984705283@qq.com](mailto:984705283@qq.com)
  * GitHub：[https://github.com/yongfeng200/izijin](https://github.com/yongfeng200/izijin)