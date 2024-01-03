from flask import Flask, request, render_template
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
import base64
from io import BytesIO

app = Flask(__name__)

# 配置参数
data_dir = 'data'  # 数据集路径
class_names = ['cat', 'dog']  # 类别名称
confidence_threshold = 0.4 # 置信度阈值

# 加载预训练的 ResNet 模型
model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
model.fc = torch.nn.Linear(512, len(class_names))
model.eval()

# 定义自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.file_list = os.listdir(self.data_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_name = self.file_list[index]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        label = 0 if img_name.startswith('cat') else 1
        return image, label

# 图像预处理函数
def transform_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.478, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

# 图像分类推理函数
def classify_image(image):
    input_tensor = transform_image(image)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        _, predicted_idx = torch.max(output, 1)
        predicted_label = class_names[predicted_idx.item()]

    # 返回 Base64 编码的图像数据
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return predicted_label, probabilities, encoded_image

# Flask 路由和视图函数
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        try:
            # 从请求中获取上传的图像文件
            image_file = request.files['image']
            # 将图像文件加载为 PIL 图像对象
            image = Image.open(image_file).convert('RGB')
            # 进行图像分类推理
            predicted_label, probabilities, encoded_image = classify_image(image)

            # 打印预测结果和概率（用于调试）
            print(f"预测标签: {predicted_label}")
            print(f"概率: {probabilities}")

            # 检查预测标签的置信度是否高于阈值
            if probabilities[class_names.index(predicted_label)] >= confidence_threshold:
                # 返回预测结果和 Base64 编码的图像数据到前端页面
                return render_template('app.html', label=predicted_label, probabilities=probabilities, uploaded_image=encoded_image)
            else:
                # 如果置信度低于阈值，指示为不确定
                return render_template('app.html', label='不确定', probabilities=probabilities, uploaded_image=encoded_image)

        except Exception as e:
            # 处理异常情况
            print(f"出现异常: {e}")
            return render_template('app.html', label='处理图像时出错', probabilities=None, uploaded_image=None)

    # 如果是 GET 请求，返回上传图像的表单页面
    return render_template('loop.html')

if __name__ == '__main__':
    app.run()
