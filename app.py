import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import precision_score, accuracy_score

st.set_page_config(page_title="แบบฟอร์มประเมินความเสี่ยงโรคหัวใจ", layout="wide")

st.markdown(
    """
    <style>
    .centered-container {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
    }
    </style>
    """, unsafe_allow_html=True
)

st.title("แบบฟอร์มประเมินความเสี่ยงโรคหัวใจ")

@st.cache_resource
def load_model():
    return joblib.load('rf_model.joblib')

@st.cache_resource
def load_test_data():
    X_test = pd.read_csv('X_test.csv')
    y_test = pd.read_csv('y_test.csv').iloc[:, 0]
    return X_test, y_test

model = load_model()
X_test, y_test = load_test_data()

with st.container():
    st.markdown('<div class="centered-container">', unsafe_allow_html=True)

    st.subheader("ป้อนข้อมูลสุขภาพของคุณ (ฟีเจอร์สำคัญเรียงตามโมเดล)")

    c1, c2, c3 = st.columns(3)
    with c1:
        cp = st.selectbox('ประเภทอาการเจ็บหน้าอก (cp)', options=[1, 2, 3, 4])
    with c2:
        trestbps = st.number_input('ความดันโลหิตขณะพัก (trestbps)', 50, 250, 120)
    with c3:
        chol = st.number_input('คอเลสเตอรอล (chol)', 100, 600, 200)

    c1, c2, c3 = st.columns(3)
    with c1:
        thalach = st.number_input('อัตราการเต้นหัวใจสูงสุด (thalach)', 60, 250, 150)
    with c2:
        exang = st.selectbox('เจ็บหน้าอกจากออกกำลังกาย (exang)', options=[0,1], format_func=lambda x: "ไม่มี" if x==0 else "มี")
    with c3:
        oldpeak = st.number_input('ST depression (oldpeak)', 0.0, 10.0, 1.0, format="%.1f")

    c1, c2 = st.columns(2)
    with c1:
        ca = st.selectbox('จำนวนเส้นเลือดใหญ่ที่มีปัญหา (ca)', options=[0,1,2,3,4])
    with c2:
        thal = st.selectbox('ภาวะธาลัสซีเมีย (thal)', options=[3, 6, 7], format_func=lambda x: {3:"ปกติ",6:"ข้อบกพร่องถาวร",7:"ข้อบกพร่องกลับคืนได้"}[x])

    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input('อายุ', 1, 120, 50)
    with c2:
        sex = st.selectbox('เพศ', options=[0, 1], format_func=lambda x: 'หญิง' if x == 0 else 'ชาย')

    if st.button('ทำนายความเสี่ยง'):
        input_data = np.array([[
            int(cp),            # cp
            float(trestbps),    # trestbps
            float(chol),        # chol
            float(thalach),     # thalach
            int(exang),         # exang
            float(oldpeak),     # oldpeak
            int(ca),            # ca
            int(thal),          # thal
            float(age),         # age
            int(sex)            # sex
        ]])

        try:
            prediction = model.predict(input_data)[0]

            y_pred_test = model.predict(X_test)
            precision = precision_score(y_test, y_pred_test)
            accuracy = accuracy_score(y_test, y_pred_test)

            if prediction == 1:
                st.error("⚠️ ผลลัพธ์: มีความเสี่ยงเป็นโรคหัวใจ")
            else:
                st.success("✅ ผลลัพธ์: ความเสี่ยงต่ำ ไม่เป็นโรคหัวใจ")

            st.markdown(f"---\n**ประสิทธิภาพของโมเดลบนชุดทดสอบ**  \n- Precision: {precision:.2f}  \n- Accuracy: {accuracy:.2f}")

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการทำนาย: {e}")

    st.markdown('</div>', unsafe_allow_html=True)
