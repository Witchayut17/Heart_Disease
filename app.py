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

        if prediction == 0:
            st.success("✅ ผลลัพธ์: ความเสี่ยงต่ำ ไม่เป็นโรคหัวใจ (Class 0)")
        else:
            st.error(f"⚠️ ผลลัพธ์: มีความเสี่ยงเป็นโรคหัวใจ (Class {prediction})")

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการทำนาย: {e}")

st.markdown("---")
st.markdown("## วิธีกรอกข้อมูลตามคลาส (ประมาณค่าเฉลี่ยฟีเจอร์แต่ละคลาส)")

st.markdown("""
- **Class 0 (ความเสี่ยงต่ำ):**
  - cp: 1 (เจ็บน้อย)
  - ca: 0-1 (เส้นเลือดใหญ่ปกติหรือมีปัญหาน้อย)
  - exang: ไม่มี (0)
  - oldpeak: ต่ำกว่า 1.0
  - thalach: สูง (150 ขึ้นไป)
  - chol (คอเลสเตอรอล): ต่ำกว่า 240 mg/dL
  - trestbps (ความดันโลหิตขณะพัก): ต่ำกว่า 130 mm Hg
  - age (อายุ): ต่ำกว่า 50 ปี
  - sex: ชาย/หญิง (ไม่มีผลเฉพาะเจาะจง)

- **Class 1 - 3 (ความเสี่ยงปานกลาง):**
  - cp: 2-3 (เจ็บปานกลาง)
  - ca: 1-3
  - exang: อาจมี (0 หรือ 1)
  - oldpeak: ประมาณ 1.0-2.5
  - thalach: กลางๆ (120-150)
  - chol: 240-300 mg/dL
  - trestbps: 130-160 mm Hg
  - age: 50-65 ปี
  - sex: ชาย/หญิง

- **Class 4 (ความเสี่ยงสูง):**
  - cp: 4 (เจ็บมาก)
  - ca: 3-4 (เส้นเลือดใหญ่มีปัญหาหนัก)
  - exang: มี (1)
  - oldpeak: สูงกว่า 2.5
  - thalach: ต่ำกว่า 120
  - chol: มากกว่า 300 mg/dL
  - trestbps: มากกว่า 160 mm Hg
  - age: มากกว่า 65 ปี
  - sex: ชาย/หญิง
""")
