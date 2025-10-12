import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="แบบฟอร์มประเมินความเสี่ยงโรคหัวใจ", layout="wide")

st.title("แบบฟอร์มประเมินความเสี่ยงโรคหัวใจ")

@st.cache_resource
def load_model():
    return joblib.load('rf_model.joblib')

model = load_model()

# แสดง class ที่โมเดลเรียนรู้
st.write(f"โมเดลรองรับคลาส: {model.classes_}")

col1, col_right = st.columns([1, 1])

with col1:
    st.subheader("ป้อนข้อมูลสุขภาพของคุณ")
    with st.form(key='heart_risk_form'):
        thal = st.selectbox('ภาวะธาลัสซีเมีย (thal)', options=[3,6,7],
                            format_func=lambda x: {3:"ปกติ",6:"ข้อบกพร่องถาวร",7:"ข้อบกพร่องกลับคืนได้"}[x])

        ca = st.selectbox('จำนวนเส้นเลือดใหญ่ที่มีปัญหา (ca)', options=[0,1,2,3,4])

        cp = st.selectbox('ประเภทอาการเจ็บหน้าอก (cp)', options=[1,2,3,4])

        oldpeak = st.selectbox('ST depression (oldpeak)', options=[
            0.0, 0.5, 1.0, 2.0, 3.0, 5.0], format_func=lambda x: (
                "< 0.5" if x == 0.0 else
                "0.5 - 1.0" if x == 0.5 else
                "1.0 - 2.0" if x == 1.0 else
                "2.0 - 3.0" if x == 2.0 else
                "3.0 - 5.0" if x == 3.0 else
                "> 5.0"
            ))

        thalach = st.selectbox('อัตราการเต้นหัวใจสูงสุด (thalach)', options=[
            100, 120, 150, 180, 200], format_func=lambda x: (
                "< 120 bpm" if x == 100 else
                "120 - 150 bpm" if x == 120 else
                "150 - 180 bpm" if x == 150 else
                "180 - 200 bpm" if x == 180 else
                "> 200 bpm"
            ))

        exang = st.selectbox('เจ็บหน้าอกจากออกกำลังกาย (exang)', options=[0,1], format_func=lambda x: "ไม่มี" if x==0 else "มี")

        age = st.selectbox('ช่วงอายุ', options=[40, 55, 70, 85], format_func=lambda x: (
            "< 50 ปี" if x == 40 else
            "50 - 65 ปี" if x == 55 else
            "65 - 85 ปี" if x == 70 else
            "> 85 ปี"
        ))

        sex = st.selectbox('เพศ', options=[0,1], format_func=lambda x: 'หญิง' if x==0 else 'ชาย')

        trestbps = st.selectbox('ความดันโลหิตขณะพัก (trestbps)', options=[
            110, 130, 150, 170, 200], format_func=lambda x: (
                "< 130 mm Hg" if x == 110 else
                "130 - 150 mm Hg" if x == 130 else
                "150 - 170 mm Hg" if x == 150 else
                "170 - 200 mm Hg" if x == 170 else
                "> 200 mm Hg"
            ))

        chol = st.selectbox('คอเลสเตอรอล (chol)', options=[
            200, 250, 300, 350, 400], format_func=lambda x: (
                "< 240 mg/dL" if x == 200 else
                "240 - 300 mg/dL" if x == 250 else
                "300 - 350 mg/dL" if x == 300 else
                "350 - 400 mg/dL" if x == 350 else
                "> 400 mg/dL"
            ))

        submit_button = st.form_submit_button(label='ทำนายความเสี่ยง')

    if submit_button:
        input_data = np.array([[ 
            int(cp), float(trestbps), float(chol), float(thalach), int(exang),
            float(oldpeak), int(ca), int(thal), float(age), int(sex)
        ]])

        st.write("ข้อมูลที่ส่งเข้าโมเดล:", input_data)

        try:
            prediction = model.predict(input_data)[0]

            if prediction == 0:
                st.success("✅ ผลลัพธ์: ความเสี่ยงต่ำ ไม่เป็นโรคหัวใจ (Class 0)")
            elif prediction == 1:
                st.warning("⚠️ ผลลัพธ์: ความเสี่ยงระดับต่ำ-ปานกลาง (Class 1)")
            elif prediction == 2:
                st.warning("⚠️ ผลลัพธ์: ความเสี่ยงระดับปานกลาง (Class 2)")
            elif prediction == 3:
                st.error("❗ ผลลัพธ์: ความเสี่ยงสูง (Class 3)")
            elif prediction == 4:
                st.error("❗ ผลลัพธ์: ความเสี่ยงสูงมาก (Class 4)")
            else:
                st.info(f"ผลลัพธ์: คลาส {prediction}")

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการทำนาย: {e}")

with col_right:
    st.subheader("คู่มือการกรอกแบบฟอร์ม")
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
            - thal: ปกติ หรือ ข้อบกพร่องเล็กน้อย
            - ca: 0-1
            - cp: 1-2 (เจ็บน้อยถึงปานกลาง)
            - oldpeak: < 1.0
            - thalach: > 150 bpm
            - exang: 0 (ไม่มี)
            - age: < 50 ปี
            - sex: ชาย/หญิง
            - trestbps: < 130 mm Hg
            - chol: < 240 mg/dL
            """)

        with col_b:
            st.markdown("""
            ### Class 1 - 3 (ความเสี่ยงปานกลาง)
            - thal: ข้อบกพร่องถาวร หรือ กลับคืนได้
            - ca: 1-3
            - cp: 2-3
            - oldpeak: 1.0-2.5
            - thalach: 120-150 bpm
            - exang: 0 หรือ 1
            - age: 50-65 ปี
            - sex: ชาย/หญิง
            - trestbps: 130-160 mm Hg
            - chol: 240-300 mg/dL
            """)

        with col_c:
            st.markdown("""
            ### Class 4 (ความเสี่ยงสูง)
            - thal: ข้อบกพร่องถาวร หรือ กลับคืนได้
            - ca: 3-4
            - cp: 4
            - oldpeak: > 2.5
            - thalach: < 120 bpm
            - exang: 1 (มี)
            - age: > 65 ปี
            - sex: ชาย/หญิง
            - trestbps: > 160 mm Hg
            - chol: > 300 mg/dL
            """)
