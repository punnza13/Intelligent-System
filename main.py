import streamlit as st

st.title("📌 SE Project: การวิเคราะห์โมเดล Machine Learning")
st.write("---")

st.markdown("""
## 🔍 เกี่ยวกับโปรเจกต์นี้
โปรเจกต์นี้เป็นการศึกษาและทดลองใช้ **Machine Learning** กับข้อมูลจริง โดยมี **2 โมเดลหลัก** ได้แก่:

1. **โมเดลวิเคราะห์เกม Tic-Tac-Toe**  
   - ใช้ **อัลกอริทึมต่างๆ** เช่น Logistic Regression, Random Forest, SVM, XGBoost  
   - ทำนายผลแพ้ชนะของเกมจากตำแหน่งบนกระดาน  
2. **โมเดลจำแนกภาพ MNIST**  
   - ใช้ **Neural Network (PyTorch)**  
   - แยกแยะตัวเลข (0-9) จากภาพเขียนลายมือ  

โปรเจกต์นี้ช่วยให้เข้าใจหลักการทำงานของ **Machine Learning, การเตรียมข้อมูล, และการฝึกโมเดล**
""")

st.write("---")
st.write("### 📂 เลือกหน้าที่ต้องการเข้าใช้งาน:")
col1, col2, col3 = st.columns(3)
with col1:
    st.page_link("pages/page1.py", label="📖 ข้อมูลโมเดล", icon="📘")
with col2:
    st.page_link("pages/page2.py", label="🎮 โมเดล Tic-Tac-Toe", icon="⭕")
with col3:
    st.page_link("pages/page3.py", label="🔢 โมเดล MNIST", icon="✏️")
