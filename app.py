import os
import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="แบบฟอร์มประเมินความเสี่ยงโรคหัวใจ", layout="wide")

# ✅ ย้าย Sidebar ไปขวาด้วย CSS
st.markdown("""
    <style>
        [data-testid="stSidebar"][aria-expanded="true"]{
            direction: rtl;
        }
        [data-testid="stSidebar"] div:first-child {
            direction: ltr;
        }
    </style>
""", unsafe_allow_html=True)

# ✅ โหลดโมเดล
@st.cache_resource
def load_model_and_scaler():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_and_scaler()

# ✅ ส่วน sidebar ขวา
with st.sidebar:
    st.header("คำอธิบายฟิลด์")
    with st.expander("ดูรายละเอียด"):
        st.markdown("""
        - **อายุ (age):** อายุของผู้ป่วย (ปี)
        - **เพศ (sex):** 0 = หญิง, 1 = ชาย
        - **อาการเจ็บหน้าอก (cp):**
            - 1 = typical angina  
            - 2 = atypical angina  
            - 3 = non-anginal pain  
            - 4 = asymptomatic
        - **trestbps:** ความดันขณะพัก (มม.ปรอท)
        - **chol:** คอเลสเตอรอลในเลือด (mg/dl)
        - **fbs:** น้ำตาลในเลือด > 120 = 1, อื่น ๆ = 0
        - **restecg:** 0-2 ค่าคลื่นไฟฟ้าหัวใจ
        - **thalach:** ชีพจรสูงสุด (bpm)
        - **exang:** เจ็บหน้าอกจากการออกกำลังกาย (1 = มี)
        - **oldpeak:** ค่า ST depression
        - **slope:** ความชันของ ST segment
        - **ca:** จำนวนเส้นเลือดใหญ่ (0–4)
        - **thal:** ประเภทธาลัสซีเมีย (3, 6, 7)
        """)

# ✅ ส่วนฟอร์มหลัก (อยู่ฝั่งซ้าย)
st.title("แบบฟอร์มประเมินความเสี่ยงโรคหัวใจ")
st.subheader("ป้อนข้อมูลสุขภาพของคุณ")

c1, c2, c3 = st.columns([2, 1, 2])
with c1:
    age = st.number_input('อายุ', 1, 120, 50)
with c2:
    sex = st.selectbox('เพศ', options=[0, 1], format_func=lambda x: 'หญิง' if x == 0 else 'ชาย')
with c3:
    cp = st.selectbox('ประเภทอาการเจ็บหน้าอก (1–4)', options=[1, 2, 3, 4])

c1, c2, c3 = st.columns([2, 2, 2])
with c1:
    trestbps = st.number_input('ความดันโลหิตขณะพัก (มม.ปรอท)', 50, 250, 120)
with c2:
    chol = st.number_input('คอเลสเตอรอลในเลือด (mg/dl)', 100, 600, 200)
with c3:
    fbs = st.selectbox('น้ำตาลในเลือดขณะอดอาหาร', options=[0, 1], format_func=lambda x: 'ไม่เกิน 120' if x == 0 else 'มากกว่า 120')

c1, c2, c3 = st.columns([2, 2, 2])
with c1:
    restecg = st.selectbox('ผลคลื่นไฟฟ้าหัวใจขณะพัก', options=[0, 1, 2])
with c2:
    thalach = st.number_input('อัตราการเต้นหัวใจสูงสุด', 60, 250, 150)
with c3:
    exang = st.selectbox('มีอาการเจ็บหน้าอกจากการออกกำลังกายหรือไม่', options=[0, 1], format_func=lambda x: 'ไม่มี' if x == 0 else 'มี')

c1, c2, c3 = st.columns([2, 2, 2])
with c1:
    oldpeak = st.number_input('ค่า ST depression จากการออกกำลังกาย', 0.0, 10.0, 1.0, format="%.1f")
with c2:
    slope = st.selectbox('ความชันของกราฟ ST ขณะออกกำลังกาย', options=[1, 2, 3])
with c3:
    ca = st.selectbox('จำนวนเส้นเลือดใหญ่ที่เห็นจากฟลูโอโรสโคปี', options=[0, 1, 2, 3, 4])

thal = st.selectbox('ภาวะธาลัสซีเมีย', options=[3, 6, 7])

if st.button('ทำนายความเสี่ยง'):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                            exang, oldpeak, slope, ca, thal]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][prediction]

    if prediction == 1:
        st.error(f"⚠️ คุณมีความเสี่ยงในการเป็นโรคหัวใจ\n\n**ความมั่นใจของโมเดล:** {proba:.2f}\nกรุณาปรึกษาแพทย์")
    else:
        st.success(f"✅ ความเสี่ยงต่ำ\n\n**ความมั่นใจของโมเดล:** {proba:.2f}\nดูแลสุขภาพให้ดีแบบนี้ต่อไปนะคะ")
