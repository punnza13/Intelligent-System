import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import os

# 📌 โหลดโมเดล PyTorch (โครงสร้างเดียวกับตอนเทรน)
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

# 📌 โหลดโมเดลที่ฝึกไว้
model = NeuralNet()

# กำหนด path ของไฟล์ที่เก็บโมเดล
model_path = os.path.join("model", "mnist_model.pth")  # ใช้ path จากโฟลเดอร์ model

# โหลดโมเดล
model.load_state_dict(torch.load(model_path))
model.eval()

# 📌 ฟังก์ชันพรีโปรเซสภาพ
def preprocess_image(image):
    image = image.convert("L")  # แปลงเป็นภาพขาวดำ
    image = ImageOps.invert(image)  # กลับสีให้ตัวเลขเป็นขาว พื้นหลังดำ
    image = image.resize((28, 28))  # ปรับขนาดเป็น 28x28
    transform = transforms.ToTensor()
    image = transform(image)
    image = image.view(-1, 28*28)  # แปลงเป็นเวกเตอร์ขนาด 28x28
    return image

# 📌 สร้างอินเทอร์เฟซ Streamlit
import streamlit as st
st.title("🎨 วาดตัวเลข 0-9 แล้วให้โมเดลทำนาย")

# 📌 วาดตัวเลขบนแคนวาส
canvas = st.file_uploader("📤 อัปโหลดรูปตัวเลขที่วาด (28x28 pixels)", type=["png", "jpg", "jpeg"])

if canvas:
    image = Image.open(canvas)
    st.image(image, caption="📷 ภาพที่อัปโหลด", use_column_width=True)

    # 📌 พรีโปรเซสภาพและทำนาย
    image_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.argmax(output, dim=1).item()

    # 📌 แสดงผลลัพธ์
    st.write(f"🔢 โมเดลทำนายว่าเป็นเลข: **{prediction}**")
