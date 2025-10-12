import streamlit as st
import pickle
import numpy as np
from sklearn.metrics import precision_score, accuracy_score
import pandas as pd

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
    with open('rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_test_data():
    X_test = pd.read_csv('X_test.csv')
    y_test = pd.read_csv('y_test.csv')
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]
    return X_test, y_test

model = load_model()
X_test, y_test = load_test_data()

with st.container():
    st.markdown('<div class="centered-container">', unsafe_allow_html=True)

    st.subheader("ป้อนข้อมูลสุขภาพของคุณ (ฟีเจอร์สำคัญเรียงก่อน)")

    c1, c2, c3 = st.columns(3)
    with c1:
        thal = st.selectbox('ภาวะธาลัสซีเมีย (thal)', options=[3, 6, 7], format_func=lambda x: {3:"ปกติ",6:"ข้อบกพร่องถาวร",7:"ข้อบกพร่องกลับคืนได้"}[x])
    with c2:
        cp = st.selectbox('ประเภทอาการเจ็บหน้าอก (cp)', options=[1, 2, 3, 4])
    with c3:
        ca = st.selectbox('จำนวนเส้นเลือดใหญ่ที่มีปัญหา (ca)', options=[0,1,2,3,4])

    c1, c2, c3 = st.columns(3)
    with c1:
        oldpeak = st.number_input('ST depression (oldpeak)', 0.0, 10.0, 1.0, format="%.1f")
    with c2:
        thalach = st.number_input('อัตราการเต้นหัวใจสูงสุด (thalach)', 60, 250, 150)
    with c3:
        exang = st.selectbox('เจ็บหน้าอกจากออกกำลังกาย (exang)', options=[0,1], format_func=lambda x: "ไม่มี" if x==0 else "มี")

    c1, c2, c3 = st.columns(3)
    with c1:
        chol = st.number_input('คอเลสเตอรอล (chol)', 100, 600, 200)
    with c2:
        trestbps = st.number_input('ความดันโลหิตขณะพัก (trestbps)', 50, 250, 120)
    with c3:
        age = st.number_input('อายุ', 1, 120, 50)

    c1, c2 = st.columns(2)
    with c1:
        sex = st.selectbox('เพศ', options=[0, 1], format_func=lambda x: 'หญิง' if x == 0 else 'ชาย')
    with c2:
        fbs = st.selectbox('น้ำตาลในเลือดขณะอดอาหาร (fbs)', options=[0, 1], format_func=lambda x: 'ไม่เกิน 120' if x == 0 else 'มากกว่า 120')

    restecg = st.selectbox('ผลคลื่นไฟฟ้าหัวใจขณะพัก (restecg)', options=[0, 1, 2])
    slope = st.selectbox('ความชันของ ST segment (slope)', options=[1, 2, 3])

    if st.button('ทำนายความเสี่ยง'):
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                                exang, oldpeak, slope, ca, thal]])
        prediction = model.predict(input_data)[0]

        y_pred_test = model.predict(X_test)
        precision = precision_score(y_test, y_pred_test)
        accuracy = accuracy_score(y_test, y_pred_test)

        if prediction == 1:
            st.error("⚠️ ผลลัพธ์: มีความเสี่ยงเป็นโรคหัวใจ")
        else:
            st.success("✅ ผลลัพธ์: ความเสี่ยงต่ำ ไม่เป็นโรคหัวใจ")

        st.markdown(f"---\n**ประสิทธิภาพของโมเดลบนชุดทดสอบ**  \n- Precision: {precision:.2f}  \n- Accuracy: {accuracy:.2f}")

    st.markdown('</div>', unsafe_allow_html=True)
