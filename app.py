import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Streamlit 기본 설정
st.set_page_config(page_title="Lizard Classification", layout="centered")
st.markdown("## 🌿 잎사귀 vs 도마뱀 🦎")
st.write("아래에서 사진을 찍거나 업로드하여 도마뱀인지 나뭇잎인지 확인해보세요! 📷")

# 모델 및 라벨 불러오기
model = load_model('keras_model_leaf.h5', compile=False)
class_names = open('labels.txt', 'r').readlines()

# UI 입력 방식 선택
st.sidebar.header("🔍 이미지 입력 방식")
input_method = st.sidebar.radio("", ["📁 파일 업로드", "📷 카메라 사용"])

if input_method == "📷 카메라 사용":
    img_file_buffer = st.camera_input("정중앙에 사물을 위치하고 촬영하세요")
else:
    img_file_buffer = st.file_uploader("(PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

# 이미지 처리 및 예측
if img_file_buffer is not None:
    st.image(img_file_buffer, caption="🔍 입력된 이미지", use_container_width=True)
    
    image = Image.open(img_file_buffer).convert('RGB')
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    with st.spinner("AI가 이미지를 분석 중입니다...⏳"):
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]
    
    # 결과 출력
    st.success(f"🤖 예측 결과: **{class_name[2:]}**")
    st.write(f"📊 신뢰도: **{confidence_score:.2%}**")
    
    # 진행 바 추가
    st.progress(float(confidence_score))
