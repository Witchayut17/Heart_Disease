import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="แบบฟอร์มประเมินความเสี่ยงโรคหัวใจ", layout="wide")

st.title("แบบฟอร์มประเมินความเสี่ยงโรคหัวใจ")

@st.cache_resource
def load_model():
    return joblib.load('rf_model.joblib')

model = load_model()

col1, col2 = st.columns([2, 1])  # ฝั่งซ้ายกว้างกว่า ฝั่งขวาเล็กกว่า

with col1:
    st.subheader("ป้อนข้อมูลสุขภาพของคุณ")
    with st.form(key='heart_risk_form'):
        thal = st.selectbox('ภาวะธาลัสซีเมีย (thal)', options=[3,6,7],
                            format_func=lambda x: {3:"ปกติ",6:"ข้อบกพร่องถาวร",7:"ข้อบกพร่องกลับคืนได้"}[x])
        cp = st.selectbox('ประเภทอาการเจ็บหน้าอก (cp)', options=[1,2,3,4])
        ca = st.selectbox('จำนวนเส้นเลือดใหญ่ที่มีปัญหา (ca)', options=[0,1,2,3,4])
        exang = st.selectbox('เจ็บหน้าอกจากออกกำลังกาย (exang)', options=[0,1], format_func=lambda x: "ไม่มี" if x==0 else "มี")
        sex = st.selectbox('เพศ', options=[0,1], format_func=lambda x: 'หญิง' if x==0 else 'ชาย')
        oldpeak = st.slider('ST depression (oldpeak)', 0.0, 10.0, 1.0, 0.1)
        thalach = st.slider('อัตราการเต้นหัวใจสูงสุด (thalach)', 60, 250, 150)
        chol = st.slider('คอเลสเตอรอล (chol)', 100, 600, 200)
        trestbps = st.slider('ความดันโลหิตขณะพัก (trestbps)', 50, 250, 120)
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

with col2:
    with st.expander("📌 ข้อมูลสรุปและวิธีกรอกข้อมูล", expanded=True):

        # แถว 1: สรุปจำนวนข้อมูลแต่ละคลาส
        st.markdown("""
        ### สรุปข้อมูลตัวอย่างตามคลาสเป้าหมาย (target)
        - Class 0: 29 ตัวอย่าง (47.54%)
        - Class 1: 12 ตัวอย่าง (19.67%)
        - Class 2: 9 ตัวอย่าง (14.75%)
        - Class 3: 7 ตัวอย่าง (11.48%)
        - Class 4: 4 ตัวอย่าง (6.56%)
        """)

        # แถว 2: ตารางฟีเจอร์ส่วนบน (5 ฟีเจอร์แรก)
        st.markdown("""
        ### สถิติฟีเจอร์ (normalized/scaled) - ส่วนที่ 1
        | ฟีเจอร์     | mean  | min   | max   | คำอธิบาย (หน่วยจริงโดยประมาณ)                  |
        |-------------|-------|-------|-------|----------------------------------------------|
        | cp          | 2.43  | 0     | 3     | ประเภทอาการเจ็บหน้าอก 0-3 (0=ต่ำสุด, 3=สูงสุด)    |
        | trestbps    | -0.13 | -1.89 | 2.33  | ความดันโลหิตขณะพัก (50-250 mm Hg)              |
        | chol        | 0.05  | -2.41 | 2.64  | คอเลสเตอรอล (100-600 mg/dL)                    |
        | thalach     | -0.09 | -2.72 | 1.43  | อัตราการเต้นหัวใจสูงสุด (60-250 bpm)            |
        | exang       | 0.28  | 0     | 1     | เจ็บหน้าอกจากออกกำลังกาย (0=ไม่มี, 1=มี)           |
        """)

        # แถว 3: ตารางฟีเจอร์ส่วนล่าง (5 ฟีเจอร์หลัง)
        st.markdown("""
        ### สถิติฟีเจอร์ (normalized/scaled) - ส่วนที่ 2
        | ฟีเจอร์     | mean  | min   | max   | คำอธิบาย (หน่วยจริงโดยประมาณ)                  |
        |-------------|-------|-------|-------|----------------------------------------------|
        | oldpeak     | 0.02  | -0.92 | 2.68  | ST depression                                 |
        | ca          | 0.75  | 0     | 3     | จำนวนเส้นเลือดใหญ่ที่มีปัญหา                    |
        | thal        | 0.93  | 0     | 2     | ภาวะธาลัสซีเมีย (0-2)                         |
        | sex         | 0.77  | 0     | 1     | เพศ (0=หญิง, 1=ชาย)                           |
        | age         | -0.16 | -2.27 | 1.83  | อายุ (1-120 ปี) (normalized)                    |
        """)

        # วิธีกรอกข้อมูลแยกต่างหาก (ถ้าต้องการเพิ่ม)
        st.markdown("""
        ### วิธีกรอกข้อมูลตามคลาส

        - **Class 0 (ความเสี่ยงต่ำ):**
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

        - **Class 1 - 3 (ความเสี่ยงปานกลาง):**
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

        - **Class 4 (ความเสี่ยงสูง):**
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
