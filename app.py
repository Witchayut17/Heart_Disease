import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="แบบฟอร์มประเมินความเสี่ยงโรคหัวใจ", layout="wide")
st.title("แบบฟอร์มประเมินความเสี่ยงโรคหัวใจ")

@st.cache_data(show_spinner=False)
def load_model():
    return joblib.load('rf_model.joblib')

model = load_model()

model = load_model()

# ค่า mean และ std จากข้อมูลสถิติ
mean_dict = {
    'cp': 2.147541,
    'trestbps': 0.075808,
    'chol': -0.187700,
    'thalach': 0.099100,
    'exang': 0.344262,
    'oldpeak': 0.128838,
    'ca': 0.622951,
    'thal': 0.967213,
    'sex': 0.754098,
    'age': -0.083161
}

std_dict = {
    'cp': 0.891367,
    'trestbps': 0.970753,
    'chol': 0.833809,
    'thalach': 0.923236,
    'exang': 0.479070,
    'oldpeak': 1.124272,
    'ca': 0.915859,
    'thal': 0.965526,
    'sex': 0.434194,
    'age': 1.063041
}

col1, col_right = st.columns([1, 1])

with col1:
    st.subheader("ป้อนข้อมูลสุขภาพของคุณ")
    with st.form(key='heart_risk_form'):
        thal = st.selectbox('ภาวะธาลัสซีเมีย (thal)', options=[3,6,7],
                            format_func=lambda x: {3:"ปกติ",6:"ข้อบกพร่องถาวร",7:"ข้อบกพร่องกลับคืนได้"}[x])

        ca = st.selectbox('จำนวนเส้นเลือดใหญ่ที่มีปัญหา (ca)', options=[0,1,2,3,4])

        cp = st.selectbox('ประเภทอาการเจ็บหน้าอก (cp)', options=[1,2,3,4])

        oldpeak = st.selectbox('ST depression (oldpeak)', options=[0.0, 0.5, 1.0, 2.0, 3.0, 5.0],
                               format_func=lambda x: (
                                   "< 0.5" if x == 0.0 else
                                   "0.5 - 1.0" if x == 0.5 else
                                   "1.0 - 2.0" if x == 1.0 else
                                   "2.0 - 3.0" if x == 2.0 else
                                   "3.0 - 5.0" if x == 3.0 else
                                   "> 5.0"
                               ))

        thalach = st.selectbox('อัตราการเต้นหัวใจสูงสุด (thalach)', options=[100, 120, 150, 180, 200],
                               format_func=lambda x: (
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

        trestbps = st.selectbox('ความดันโลหิตขณะพัก (trestbps)', options=[110, 130, 150, 170, 200],
                                format_func=lambda x: (
                                    "< 130 mm Hg" if x == 110 else
                                    "130 - 150 mm Hg" if x == 130 else
                                    "150 - 170 mm Hg" if x == 150 else
                                    "170 - 200 mm Hg" if x == 170 else
                                    "> 200 mm Hg"
                                ))

        chol = st.selectbox('คอเลสเตอรอล (chol)', options=[200, 250, 300, 350, 400],
                            format_func=lambda x: (
                                "< 240 mg/dL" if x == 200 else
                                "240 - 300 mg/dL" if x == 250 else
                                "300 - 350 mg/dL" if x == 300 else
                                "350 - 400 mg/dL" if x == 350 else
                                "> 400 mg/dL"
                            ))

        submit_button = st.form_submit_button(label='ทำนายความเสี่ยง')

    if submit_button:
        # เก็บค่าที่ป้อน (raw values)
        raw_input = {
            'cp': int(cp),
            'trestbps': float(trestbps),
            'chol': float(chol),
            'thalach': float(thalach),
            'exang': int(exang),
            'oldpeak': float(oldpeak),
            'ca': int(ca),
            'thal': int(thal),
            'sex': int(sex),
            'age': float(age)
        }

        # แปลงเป็น standardized ตาม mean และ std
        input_standardized = []
        for feat in ['cp','trestbps','chol','thalach','exang','oldpeak','ca','thal','sex','age']:
            val = raw_input[feat]
            mean_val = mean_dict[feat]
            std_val = std_dict[feat]
            std_value = (val - mean_val) / std_val
            input_standardized.append(std_value)

        input_data = np.array([input_standardized])

        try:
            prediction = model.predict(input_data)[0]

            # Mapping class ใหม่
            if prediction == 0:
                new_class = 0
            elif prediction in [1, 2]:
                new_class = 1
            else:
                new_class = 2

            st.write("### ผลลัพธ์การทำนาย")
            if new_class == 0:
                st.success("✅ ผลลัพธ์: ความเสี่ยงต่ำ ไม่เป็นโรคหัวใจ (Class 0)")
            elif new_class == 1:
                st.warning("⚠️ ผลลัพธ์: มีความเสี่ยงปานกลาง เป็นโรคหัวใจ (Class 1)")
            else:
                st.error("🔴 ผลลัพธ์: มีความเสี่ยงสูง เป็นโรคหัวใจ (Class 2)")

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการทำนาย: {e}")

with col_right:
    st.subheader("คู่มือการกรอกแบบฟอร์ม")
    with st.expander("📌 ข้อมูลสรุปและวิธีกรอกข้อมูล", expanded=True):
        st.markdown("""
        ### สรุปข้อมูลตัวอย่างตามคลาสเป้าหมาย (target)
        - Class 0: 33 ตัวอย่าง (54.10%)
        - Class 1: 18 ตัวอย่าง (รวมคลาส 1 และ 2 เดิม)
        - Class 2: 10 ตัวอย่าง (รวมคลาส 3 และ 4 เดิม)
        """)

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown("""
            ### Class 0 (ความเสี่ยงต่ำ)
            - thal: ปกติ หรือ ข้อบกพร่องเล็กน้อย  
            - ca: 0
            - cp: 1-2  
            - oldpeak: < 1.0  
            - thalach: > 150 bpm  
            - exang: ไม่มี  
            - age: < 50 ปี  
            - trestbps: < 130 mm Hg  
            - chol: < 240 mg/dL  
            """)

        with col_b:
            st.markdown("""
            ### Class 1 (ความเสี่ยงปานกลาง)
            - thal: ข้อบกพร่องถาวร / กลับคืนได้  
            - ca: 1-2  
            - cp: 2-3  
            - oldpeak: 1.0 - 2.5  
            - thalach: 120 - 150 bpm  
            - exang: มี/ไม่มี  
            - age: 50 - 65 ปี  
            - trestbps: 130 - 160 mm Hg  
            - chol: 240 - 300 mg/dL  
            """)

        with col_c:
            st.markdown("""
            ### Class 2 (ความเสี่ยงสูง)
            - รวมกลุ่มผู้เคยอยู่ใน class 3 และ 4  
            - thal: ข้อบกพร่องถาวร / กลับคืนได้  
            - ca: 3  
            - cp: 3-4  
            - oldpeak: > 2.5  
            - thalach: < 120 bpm  
            - exang: มี  
            - age: > 65 ปี  
            - trestbps: > 160 mm Hg
            - chol: > 300 mg/dL  
            """)
