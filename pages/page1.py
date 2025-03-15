import streamlit as st

st.markdown("""
    <style>
        body {
            background-color: black;
            color: white;
        }
        pre, code {
            background-color: black !important;
            color: #00FF00 !important;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📚 รายละเอียดเกี่ยวกับโมเดลที่ใช้")

st.write("---")

st.header("📂 ที่มาของข้อมูลที่ใช้ในโปรเจกต์")

st.subheader("🧩 Tic-Tac-Toe Dataset")
st.write("""
ชุดข้อมูล Tic-Tac-Toe มาจาก [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/101/tic+tac+toe+endgame)  
ข้อมูลนี้แสดงถึงบอร์ดเกม Tic-Tac-Toe ที่มีผลแพ้ชนะชัดเจน  
แต่ละแถวในชุดข้อมูลแทนหนึ่งสถานการณ์ของเกม โดยมีคุณลักษณะดังนี้:
""")

st.code("""
top-left, top-middle, top-right, middle-left, middle-middle, middle-right, bottom-left, bottom-middle, bottom-right, class
x, x, o, x, o, o, o, x, x, positive
o, x, o, x, o, x, x, o, x, negative
...
""", language="plaintext")

st.subheader("🔢 MNIST Dataset")
st.write("""
ชุดข้อมูล MNIST เป็นชุดข้อมูลรูปภาพตัวเลขขนาด 28x28 พิกเซล  
ประกอบด้วยตัวเลข 0-9 ที่เขียนด้วยลายมือ มาจาก [Kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)  
โดยแต่ละแถวใน dataset มีโครงสร้างดังนี้:
""")

st.code("""
label, pixel1, pixel2, ..., pixel784
5, 0, 0, ..., 255
0, 34, 0, ..., 128
...
""", language="plaintext")

st.write("---")

st.header("⚙️ การเตรียมข้อมูล")

st.subheader("🔹 Tic-Tac-Toe")
st.write("""
1. แปลงค่าจากตัวอักษร ('x', 'o', 'b') เป็นตัวเลข (0, 1, 2)
2. แปลงค่าผลลัพธ์ ('positive', 'negative') เป็น (1, 0)
3. แบ่งข้อมูลเป็นชุดฝึก (80%) และชุดทดสอบ (20%)
""")

st.code("""
# แปลงตัวอักษรเป็นตัวเลข
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df.iloc[:, :-1] = df.iloc[:, :-1].apply(le.fit_transform)

# แปลงค่า y เป็น 0 และ 1
df['class'] = le.fit_transform(df['class'])
""", language="python")

st.subheader("🔹 MNIST")
st.write("""
1. แปลงค่าพิกเซลจาก 0-255 ให้เป็นค่า 0-1 (Normalization)
2. แบ่งข้อมูลเป็นชุดฝึก (80%) และชุดทดสอบ (20%)
3. แปลงเป็น Tensor สำหรับใช้ใน PyTorch
""")

st.code("""
# แปลงข้อมูลเป็น Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
""", language="python")

st.write("---")

st.header("🧠 ทฤษฎีของอัลกอริทึมที่ใช้")

st.subheader("📌 Logistic Regression")
st.write("""
**แนวคิด**: เป็นโมเดลที่ใช้สมการเส้นตรงและฟังก์ชัน sigmoid  
เพื่อแปลงค่าให้อยู่ในช่วง 0 ถึง 1 แล้วนำไปใช้พยากรณ์  
เหมาะกับปัญหาจำแนกประเภทแบบ Binary (เช่น Tic-Tac-Toe)
""")

st.subheader("📌 Random Forest")
st.write("""
**แนวคิด**: ใช้หลายๆ ต้นไม้ (Decision Tree) แล้วรวมผลลัพธ์กัน  
ช่วยลด overfitting และเพิ่มความแม่นยำ  
เป็นหนึ่งในโมเดลที่ใช้ใน Tic-Tac-Toe
""")

st.subheader("📌 Neural Network (MLP)")
st.write("""
**แนวคิด**: ใช้โครงข่ายประสาทเทียมที่มีชั้นซ่อนหลายชั้น  
เหมาะสำหรับปัญหาที่ซับซ้อน เช่น การรู้จำตัวเลข (MNIST)
""")

st.subheader("🔥 Gradient Boosting (GB)")
st.write("""
**แนวคิด**: Gradient Boosting เป็นอัลกอริทึมที่ใช้แนวคิดของ Boosting 
ซึ่งหมายถึงการรวมโมเดลหลายตัวเข้าด้วยกันเพื่อสร้างโมเดลที่แข็งแกร่งขึ้น โดย Gradient Boosting 
จะสร้างโมเดลแบบ Sequential (ทีละตัว) และปรับปรุงข้อผิดพลาดของโมเดลก่อนหน้าไปเรื่อย ๆ
""")

st.write("---")

st.header("🔬 กระบวนการพัฒนาโมเดล")

st.subheader("🧩 Tic-Tac-Toe")
st.write("""
1. ใช้อัลกอริทึม Logistic Regression, Random Forest, SVM, XGBoost ฯลฯ  
2. ใช้ `cross-validation` เพื่อเลือกโมเดลที่ดีที่สุด  
3. ประเมินความแม่นยำของแต่ละโมเดล
""")

st.code("""
from sklearn.model_selection import cross_val_score
rf_scores = cross_val_score(RandomForestClassifier(), X, y, cv=5)
print(f"Random Forest Accuracy: {rf_scores.mean():.4f}")
""", language="python")

st.subheader("🔢 MNIST")
st.write("""
1. ใช้โครงข่ายประสาทเทียม (Neural Network) ที่มี 2 ชั้น  
2. ใช้ฟังก์ชัน `ReLU` และ `Softmax` ในชั้นสุดท้าย  
3. ใช้ `Adam Optimizer` เพื่อเร่งความเร็วในการฝึกโมเดล  
4. ฝึกโมเดลเป็นเวลา 10 epochs
""")

st.code("""
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
""", language="python")

st.write("---")

st.header("📈 ผลลัพธ์ของโมเดล")

st.subheader("🧩 Tic-Tac-Toe")
st.write("""
- Logistic Regression Accuracy: **0.6927**
- Random Forest Accuracy: **0.9323**
- SVM Accuracy: 0.8802
- Gradient Boosting Accuracy: **0.9271**
- XGBoost Accuracy: **0.8906**
- AdaBoost Accuracy: **0.7552**
- Random Forest Cross-Validation Accuracy: **0.7977**
- SVM Cross-Validation Accuracy: **0.7956**
- Gradient Boosting Cross-Validation Accuracy: **0.8228**
- XGBoost Cross-Validation Accuracy: **0.8123**
- AdaBoost Cross-Validation Accuracy: **0.7014**
**ที่เลือกใช้**: Random Forest, Gradient Boosting, XGBoost,K-Nearest Neighbors,Decision Tree เพราะมีความแม่นยำสูง และเขียนง่าย
""")

st.subheader("🔢 MNIST")
st.write("""
- Neural Network ให้ความแม่นยำ **98.89%** บนชุดทดสอบ
""")

st.write("---")

## 🎯 **สรุป**
st.header("🎯 สรุป")
st.write("""
- Tic-Tac-Toe ใช้โมเดลคลาสสิคเช่น Logistic Regression, SVM, XGBoost  
- MNIST ใช้ Neural Network ที่ซับซ้อนขึ้น  
- Random Forest และ Neural Network เป็นโมเดลที่ดีที่สุดสำหรับปัญหานี้
""")

st.markdown("""## โดยรวมตัวโมเดลผมได้เามาจาก Assingment ต่าง กับ chatGPT+DeepSeek ส่วนตัวของการ train model ผมก็ไปทำใน Google Colab แล้วโหลดออกมาเป็นไฟล์ .pth แล้วนำมาใช้ใน streamlit ครับ""")
st.markdown("""## ผมไม่รู้คนอื่นเป็นไหมเเต่ผมใช้ Tensorflow ไม่ได้เลยใช้ Pytorch ครับ""")

