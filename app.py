import streamlit as st
import requests
import json
import time

# === Configuration ===
API_KEY = ""  # 🔐 Replace with your Vertex AI API Key
PROJECT_ID = "video-generation-461605"
LOCATION = "us-central1"
MODEL_ID = "veo-2.0-generate-001"
API_ENDPOINT = f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_ID}:predictLongRunning"

# === Streamlit UI ===
st.set_page_config(page_title="🎬 Veo 2.0 Video Generator", layout="centered")
st.title("🎬 Veo 2.0 Text-to-Video Generator")
st.markdown("Generate cinematic videos using **Google Veo 2.0** with a simple prompt!")

# === Input Form ===
with st.form("video_form"):
    prompt = st.text_area("Prompt", "A cinematic shot of a robot playing guitar on Mars at sunset")
    duration = st.slider("Duration (seconds)", min_value=2, max_value=10, value=6)
    submitted = st.form_submit_button("🚀 Generate Video")

# === Helper Functions ===
def poll_operation(operation_name):
    poll_url = f"https://{LOCATION}-aiplatform.googleapis.com/v1/{operation_name}"
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }
    while True:
        poll_response = requests.get(poll_url, headers=headers)
        result = poll_response.json()
        if result.get("done"):
            return result
        time.sleep(5)

# === Trigger Video Generation ===
if submitted:
    st.info("📡 Sending request to Google Veo model...")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "endpoint": f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_ID}",
        "instances": [
            {"prompt": prompt + "\n"}
        ],
        "parameters": {
            "aspectRatio": "16:9",
            "sampleCount": 1,
            "durationSeconds": str(duration),
            "personGeneration": "allow_adult",
            "enablePromptRewriting": True,
            "addWatermark": True,
            "includeRaiReason": True
        }
    }

    response = requests.post(API_ENDPOINT, headers=headers, json=payload)

    if response.status_code == 200:
        operation = response.json()
        operation_name = operation.get("name")
        st.success("✅ Request submitted successfully.")
        st.code(operation_name, language="text")

        with st.spinner("⏳ Waiting for video generation to complete..."):
            result = poll_operation(operation_name)

        try:
            video_url = result["response"]["predictions"][0]["videoUri"]
            st.video(video_url)
            st.success("🎉 Video generated successfully!")
            st.markdown(f"[🔗 Download Video]({video_url})")
        except Exception as e:
            st.error("❌ Failed to extract video URL.")
            st.json(result)
    else:
        st.error(f"❌ API Error {response.status_code}")
        st.json(response.json())
