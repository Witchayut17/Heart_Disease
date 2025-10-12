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

        # ข้อความคำอธิบายแต่ละคลาส
        class_dict = {
            0: "ความเสี่ยงต่ำ ไม่เป็นโรคหัวใจ",
            1: "ความเสี่ยงระดับต่ำถึงปานกลาง",
            2: "ความเสี่ยงระดับปานกลาง",
            3: "ความเสี่ยงระดับสูง",
            4: "ความเสี่ยงสูงมาก"
        }

        result_text = class_dict.get(prediction, "ไม่ทราบคลาส")

        if prediction == 0:
            st.success(f"✅ ผลลัพธ์: เป็นคลาส {prediction} — {result_text}")
        else:
            st.error(f"⚠️ ผลลัพธ์: เป็นคลาส {prediction} — {result_text}")

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
  - chol, trestbps, age: อยู่ในช่วงปกติ
  
- **Class 1 - 3 (ความเสี่ยงปานกลาง):**
  - cp: 2-3
  - ca: 1-3
  - exang: อาจมี (0 หรือ 1)
  - oldpeak: ประมาณ 1-2.5
  - thalach: กลางๆ (120-150)
  - chol, trestbps, age: ค่อยๆเพิ่มขึ้น
  
- **Class 4 (ความเสี่ยงสูง):**
  - cp: 4 (เจ็บมาก)
  - ca: 3-4 (เส้นเลือดใหญ่มีปัญหาหนัก)
  - exang: มี (1)
  - oldpeak: สูงกว่า 2.5
  - thalach: ต่ำกว่า 120
  - chol, trestbps, age: สูงกว่าเฉลี่ยมาก
""")
