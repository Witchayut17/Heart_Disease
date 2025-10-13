import streamlit as st
import numpy as np
import joblib
import pandas as pd

st.set_page_config(page_title="แบบฟอร์มประเมินความเสี่ยงโรคหัวใจ", layout="wide")
st.title("แบบฟอร์มประเมินความเสี่ยงโรคหัวใจ")

@st.cache_resource
def load_model():
    return joblib.load('rf_model.joblib')

model = load_model()

@st.cache_data
def load_data():
    df = pd.read_csv('processed_final.csv')
    df['target'] = df['target'].apply(lambda x: 2 if x in [3,4] else 1 if x in [1,2] else 0)
    counts = df['target'].value_counts().sort_index()
    total = counts.sum()
    percent_0 = counts.get(0, 0) / total * 100
    percent_1 = counts.get(1, 0) / total * 100
    percent_2 = counts.get(2, 0) / total * 100
    return counts, (percent_0, percent_1, percent_2)

counts, (percent_0, percent_1, percent_2) = load_data()

col1, col_right = st.columns([1, 2])

with col1:
    st.subheader("ป้อนข้อมูลสุขภาพของคุณ")
    with st.form(key='heart_risk_form'):
        thal = st.selectbox('ภาวะธาลัสซีเมีย (thal)', options=[3,6,7],
                            format_func=lambda x: {3:"ปกติ",6:"เป็นถาวร",7:"เป็นแต่หายได้"}[x])

        ca = st.selectbox('จำนวนเส้นเลือดใหญ่ที่มีปัญหา(ca)', options=[0,1,2,3])

        cp = st.selectbox('ประเภทอาการเจ็บหน้าอก(cp)', options=[1,2,3,4])

        oldpeak = st.selectbox('ST depression(oldpeak)', options=[0.0, 0.5, 1.0, 2.0, 3.0, 5.0],
                               format_func=lambda x: (
                                   "< 0.5" if x == 0.0 else
                                   "0.5 - 1.0" if x == 0.5 else
                                   "1.0 - 2.0" if x == 1.0 else
                                   "2.0 - 3.0" if x == 2.0 else
                                   "3.0 - 5.0" if x == 3.0 else
                                   "> 5.0"
                               ))

        thalach = st.selectbox('อัตราการเต้นหัวใจสูงสุด(thalach)', options=[100, 120, 150, 180, 200],
                               format_func=lambda x: (
                                   "< 120 bpm" if x == 100 else
                                   "120 - 150 bpm" if x == 120 else
                                   "150 - 180 bpm" if x == 150 else
                                   "180 - 200 bpm" if x == 180 else
                                   "> 200 bpm"
                               ))

        exang = st.selectbox('เจ็บหน้าอกจากออกกำลังกาย(exang)', options=[0,1], format_func=lambda x: "ไม่มี" if x==0 else "มี")

        age = st.selectbox('ช่วงอายุ', options=[40, 55, 70, 85], format_func=lambda x: (
            "< 50 ปี" if x == 40 else
            "50 - 65 ปี" if x == 55 else
            "65 - 85 ปี" if x == 70 else
            "> 85 ปี"
        ))

        sex = st.selectbox('เพศ', options=[0,1], format_func=lambda x: 'หญิง' if x==0 else 'ชาย')

        trestbps = st.selectbox('ความดันโลหิตขณะพัก(trestbps)', options=[110, 130, 150, 170, 200],
                                format_func=lambda x: (
                                    "< 130 mm Hg" if x == 110 else
                                    "130 - 150 mm Hg" if x == 130 else
                                    "150 - 170 mm Hg" if x == 150 else
                                    "170 - 200 mm Hg" if x == 170 else
                                    "> 200 mm Hg"
                                ))

        chol = st.selectbox('คอเลสเตอรอล(chol)', options=[200, 250, 300, 350, 400],
                            format_func=lambda x: (
                                "< 240 mg/dL" if x == 200 else
                                "240 - 300 mg/dL" if x == 250 else
                                "300 - 350 mg/dL" if x == 300 else
                                "350 - 400 mg/dL" if x == 350 else
                                "> 400 mg/dL"
                            ))

        submit_button = st.form_submit_button(label='ทำนายความเสี่ยง')

    if submit_button:
        input_data = np.array([[int(cp), float(trestbps), float(chol), float(thalach), int(exang),
                                float(oldpeak), int(ca), int(thal), float(age), int(sex)]])

        try:
            proba = model.predict_proba(input_data)[0]

            new_class = int(np.argmax(proba))

            st.write("ผลลัพธ์การทำนาย")
            if new_class == 0:
                st.success("ผลลัพธ์: ความเสี่ยงต่ำ ไม่เป็นโรคหัวใจ (Class 0)")
            elif new_class == 1:
                st.warning("ผลลัพธ์: มีความเสี่ยงปานกลาง เป็นโรคหัวใจ (Class 1)")
            else:
                st.error("ผลลัพธ์: มีความเสี่ยงสูง เป็นโรคหัวใจ (Class 2)")

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการทำนาย: {e}")

with col_right:
    st.subheader("คู่มือการกรอกแบบฟอร์ม")
    with st.expander("ข้อมูลสรุปและวิธีกรอกข้อมูล", expanded=True):
        st.markdown(f"""
        ### สรุปข้อมูลตัวอย่างตามคลาสเป้าหมาย (target)
        - Class 0: {counts.get(0, 0)} ตัวอย่าง ({percent_0:.2f}%)
        - Class 1: {counts.get(1, 0)} ตัวอย่าง ({percent_1:.2f}%)
        - Class 2: {counts.get(2, 0)} ตัวอย่าง ({percent_2:.2f}%)
        """)

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown("""
            ### Class 0 (ความเสี่ยงต่ำ)
            - thal (ภาวะธาลัสซีเมีย): ปกติ
            - ca (จำนวนเส้นเลือดใหญ่ที่ตีบ): 0  
            - cp (ประเภทอาการเจ็บหน้าอก): 1-2  
                - 1: เจ็บแน่นหน้าอกแบบ typical angina  
                - 2: เจ็บแน่นหน้าอกแบบ atypical angina  
            - oldpeak (ระดับ ST depression): < 1.0  
                - ค่าที่ลดลงของ ST segment ขณะออกกำลังกาย ยิ่งต่ำยิ่งดี  
            - thalach (อัตราการเต้นหัวใจสูงสุด): > 150 bpm  
            - exang (อาการเจ็บหน้าอกจากการออกกำลังกาย): ไม่มี  
            - age (อายุ): < 50 ปี  
            - trestbps (ความดันโลหิตขณะพัก): < 130 mm Hg  
            - chol (คอเลสเตอรอล): < 240 mg/dL  
            """)

        with col_b:
            st.markdown("""
            ### Class 1 (ความเสี่ยงปานกลาง)
            - thal (ภาวะธาลัสซีเมีย): เป็นถาวร/หายได้
            - ca (จำนวนเส้นเลือดใหญ่ที่ตีบ): 1-2  
            - cp (ประเภทอาการเจ็บหน้าอก): 2-3  
                - 2: เจ็บแน่นหน้าอกแบบ atypical angina  
                - 3: เจ็บหน้าอกแบบไม่เกี่ยวกับหัวใจ
            - oldpeak (ระดับ ST depression): 1.0 - 2.5
            - thalach (อัตราการเต้นหัวใจสูงสุด): 120 - 150 bpm
            - exang (อาการเจ็บหน้าอกจากการออกกำลังกาย): มี หรือ ไม่มี
            - age (อายุ): 50 - 65 ปี
            - trestbps (ความดันโลหิตขณะพัก): 130 - 160 mm Hg  
            - chol (คอเลสเตอรอล): 240 - 300 mg/dL 
            """)

        with col_c:
            st.markdown("""
            ### Class 2 (ความเสี่ยงสูง)
            - thal (ภาวะธาลัสซีเมีย): เป็นถาวร
            - ca (จำนวนเส้นเลือดใหญ่ที่ตีบ): 3  
            - cp (ประเภทอาการเจ็บหน้าอก): 3-4  
                - 3: เจ็บหน้าอกแบบไม่เกี่ยวกับหัวใจ
                - 4: เจ็บหน้าอกแบบรุนแรง
            - oldpeak (ระดับ ST depression): > 2.5  
            - thalach (อัตราการเต้นหัวใจสูงสุด): < 120 bpm  
            - exang (อาการเจ็บหน้าอกจากการออกกำลังกาย): มี  
            - age (อายุ): > 65 ปี  
            - trestbps (ความดันโลหิตขณะพัก): > 160 mm Hg  
            - chol (คอเลสเตอรอล): > 300 mg/dL  
            """)


