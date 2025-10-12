import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="แบบฟอร์มประเมินความเสี่ยงโรคหัวใจ", layout="wide")

st.title("แบบฟอร์มประเมินความเสี่ยงโรคหัวใจ")

@st.cache_resource
def load_model():
    return joblib.load('rf_model.joblib')

model = load_model()

# กำหนดคอลัมน์ ฝั่งซ้าย(ฟอร์ม) และขวา (ข้อมูลสรุป)
col1, col_right = st.columns([1, 1])

with col1:
    st.subheader("ป้อนข้อมูลสุขภาพของคุณ")
    with st.form(key='heart_risk_form'):
        # inline แถวแรก dropdown เล็กๆ รวมกัน
        col_cp, col_ca, col_exang, col_sex = st.columns([1,1,1,1])
        with col_cp:
            cp = st.selectbox('ประเภทอาการเจ็บหน้าอก (cp)', options=[1,2,3,4])
        with col_ca:
            ca = st.selectbox('จำนวนเส้นเลือดใหญ่ที่มีปัญหา (ca)', options=[0,1,2,3,4])
        with col_exang:
            exang = st.selectbox('เจ็บหน้าอกจากออกกำลังกาย (exang)', options=[0,1], format_func=lambda x: "ไม่มี" if x==0 else "มี")
        with col_sex:
            sex = st.selectbox('เพศ', options=[0,1], format_func=lambda x: 'หญิง' if x==0 else 'ชาย')

        # แถวถัดมา 2 ตัว slider inline
        col_oldpeak, col_thalach = st.columns(2)
        with col_oldpeak:
            oldpeak = st.slider('ST depression (oldpeak)', 0.0, 10.0, 1.0, 0.1)
        with col_thalach:
            thalach = st.slider('อัตราการเต้นหัวใจสูงสุด (thalach)', 60, 250, 150)

        # แถวถัดมา 2 ตัว slider inline
        col_chol, col_trestbps = st.columns(2)
        with col_chol:
            chol = st.slider('คอเลสเตอรอล (chol)', 100, 600, 200)
        with col_trestbps:
            trestbps = st.slider('ความดันโลหิตขณะพัก (trestbps)', 50, 250, 120)

        # แถวสุดท้าย อายุกับ thal inline
        col_age, col_thal = st.columns(2)
        with col_thal:
            thal = st.selectbox('ภาวะธาลัสซีเมีย (thal)', options=[3,6,7],
                                format_func=lambda x: {3:"ปกติ",6:"ข้อบกพร่องถาวร",7:"ข้อบกพร่องกลับคืนได้"}[x])
        with col_age:
            age = st.slider('อายุ', 1, 120, 50)

        submit_button = st.form_submit_button(label='ทำนายความเสี่ยง')


    if submit_button:
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

with col_right:
    with st.expander("📌 ข้อมูลสรุปและวิธีกรอกข้อมูล", expanded=True):
        st.markdown("""
        ### สรุปข้อมูลตัวอย่างตามคลาสเป้าหมาย (target)
        - Class 0: 29 ตัวอย่าง (47.54%)
        - Class 1: 12 ตัวอย่าง (19.67%)
        - Class 2: 9 ตัวอย่าง (14.75%)
        - Class 3: 7 ตัวอย่าง (11.48%)
        - Class 4: 4 ตัวอย่าง (6.56%)
        """)

        col_a, col_b, col_c = st.columns([1, 1, 1])

        with col_a:
            st.markdown("""
            ### Class 0 (ความเสี่ยงต่ำ)
            - cp: 1-2 (เจ็บน้อยถึงปานกลาง)
            - ca: 0-1
            - exang: 0 (ไม่มี)
            - oldpeak: < 1.0
            - thalach: > 150 bpm
            - chol: < 240 mg/dL
            - trestbps: < 130 mm Hg
            - age: < 50 ปี
            - sex: ชาย/หญิง
            - thal: ปกติ หรือ ข้อบกพร่องเล็กน้อย (3 หรือ 6)
            """)

        with col_b:
            st.markdown("""
            ### Class 1 - 3 (ความเสี่ยงปานกลาง)
            - cp: 2-3
            - ca: 1-3
            - exang: 0 หรือ 1
            - oldpeak: 1.0-2.5
            - thalach: 120-150 bpm
            - chol: 240-300 mg/dL
            - trestbps: 130-160 mm Hg
            - age: 50-65 ปี
            - sex: ชาย/หญิง
            - thal: ข้อบกพร่องถาวร หรือ กลับคืนได้ (6 หรือ 7)
            """)

        with col_c:
            st.markdown("""
            ### Class 4 (ความเสี่ยงสูง)
            - cp: 4
            - ca: 3-4
            - exang: 1 (มี)
            - oldpeak: > 2.5
            - thalach: < 120 bpm
            - chol: > 300 mg/dL
            - trestbps: > 160 mm Hg
            - age: > 65 ปี
            - sex: ชาย/หญิง
            - thal: ข้อบกพร่องถาวร หรือ กลับคืนได้ (6 หรือ 7)
            """)
