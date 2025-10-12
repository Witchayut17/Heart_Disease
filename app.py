import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="แบบฟอร์มประเมินความเสี่ยงโรคหัวใจ", layout="wide")

st.title("แบบฟอร์มประเมินความเสี่ยงโรคหัวใจ")

@st.cache_resource
def load_model():
    return joblib.load('rf_model.joblib')

model = load_model()

st.subheader("ป้อนข้อมูลสุขภาพของคุณ")

col1, col2, col3 = st.columns(3)

with col1:
    thal = st.selectbox('ภาวะธาลัสซีเมีย (thal)', options=[3,6,7],
                        format_func=lambda x: {3:"ปกติ",6:"ข้อบกพร่องถาวร",7:"ข้อบกพร่องกลับคืนได้"}[x])
    cp = st.selectbox('ประเภทอาการเจ็บหน้าอก (cp)', options=[1,2,3,4])
    ca = st.selectbox('จำนวนเส้นเลือดใหญ่ที่มีปัญหา (ca)', options=[0,1,2,3,4])
    exang = st.selectbox('เจ็บหน้าอกจากออกกำลังกาย (exang)', options=[0,1], format_func=lambda x: "ไม่มี" if x==0 else "มี")
    sex = st.selectbox('เพศ', options=[0,1], format_func=lambda x: 'หญิง' if x==0 else 'ชาย')

with col2:
    oldpeak = st.slider('ST depression (oldpeak)', 0.0, 10.0, 1.0, 0.1)
    thalach = st.slider('อัตราการเต้นหัวใจสูงสุด (thalach)', 60, 250, 150)
    chol = st.slider('คอเลสเตอรอล (chol)', 100, 600, 200)

with col3:
    trestbps = st.slider('ความดันโลหิตขณะพัก (trestbps)', 50, 250, 120)
    age = st.slider('อายุ', 1, 120, 50)

if st.button('ทำนายความเสี่ยง'):
    input_data = np.array([[
        int(cp), float(trestbps), float(chol), float(thalach), int(exang),
        float(oldpeak), int(ca), int(thal), float(age), int(sex)
    ]])

    try:
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.error("⚠️ ผลลัพธ์: มีความเสี่ยงเป็นโรคหัวใจ")
        else:
            st.success("✅ ผลลัพธ์: ความเสี่ยงต่ำ ไม่เป็นโรคหัวใจ")

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการทำนาย: {e}")
