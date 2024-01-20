import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import json
import base64

# 画像をアップロードして予測を取得する関数
def get_prediction(file, anomaly_score_threshold):
    url = "http://127.0.0.1:8000/predict"

    # ファイルを BytesIO に変換
    file_bytes = BytesIO(file.getvalue())

    # ファイルをリクエストの一部として送信
    files = {"file": ("image.jpg", file_bytes, "image/jpeg")}

    # しきい値をクエリパラメータとして追加
    params = {"anomaly_score_threshold": anomaly_score_threshold}
    
    response = requests.post(url, files=files, params=params)

    # レスポンスが成功したか確認
    if response.status_code == 200:
        # レスポンス内容を JSON として解析
        try:
            json_data = json.loads(response.text)
            print("Debug: JSON Response:", json_data)  # Add this line for debugging

            # Add the following line to display the entire response
            print("Debug: Full Response Text:", response.text)

            return json_data
        except json.JSONDecodeError as e:
            st.error(f"JSON の解析エラー: {e}")
            st.error(f"レスポンステキスト: {response.text}")
            return {"error": "無効な JSON レスポンス"}
    else:
        st.error(f"エラー {response.status_code}: {response.text}")
        st.write("レスポンステキスト:", response.text)  # Add this line to display response text
        return {"error": f"サーバーエラー {response.status_code}"}

# Streamlit app
def main():
    st.title("Cookie Defect Detection App")
    st.subheader("Upload an image to check for defects.")
    
    # Sidebar
    anomaly_score_threshold = st.sidebar.slider(
        "Anomaly Score Threshold", min_value=0.0, max_value=0.1, value=0.027, step=0.0001, format="%.4f",
    )
    st.sidebar.markdown("### Instructions")
    st.sidebar.write("1. Upload an image.")
    st.sidebar.write("2. Check the defect detection results.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        # Get prediction from FastAPI server with the specified anomaly_score_threshold
        prediction = get_prediction(uploaded_file, anomaly_score_threshold)  # ここでしきい値を渡す

        # Display visualizations in 1 row and 3 columns
        st.write("### Visualizations")
        col1, col2, col3 = st.columns(3)

        # Display the original image in the first column
        col1.image(Image.open(BytesIO(base64.b64decode(prediction['visualizations']['original']))), caption="Original Image", use_column_width=True)

        # Display the reconstructed image in the second column
        col2.image(Image.open(BytesIO(base64.b64decode(prediction['visualizations']['reconstructed']))), caption="Reconstructed Image", use_column_width=True)

        # Display the overlaid image with heatmap in the third column
        col3.image(Image.open(BytesIO(base64.b64decode(prediction['visualizations']['difference']))), caption="Overlaid Image with Heatmap", use_column_width=True)
        
        # Display prediction results
        st.write("### Prediction Results")
        if prediction['is_defective']:
            st.markdown(f"- **Defect Status:** <span style='color:red'>Defective</span>", unsafe_allow_html=True)
        else:
            st.write("- **Defect Status:** Not Defective")
        st.write(f"- **Anomaly Score:** {prediction['anomaly_score']:.4f}")

        # Print debug information
        print("Debug: JSON Response:", prediction)  # Add this line for debugging

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
