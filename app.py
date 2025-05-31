import streamlit as st
from diffusers import StableVideoDiffusionPipeline, StableDiffusionPipeline
import torch
from PIL import Image
import imageio
import tempfile
import os
from huggingface_hub import login

st.set_page_config(page_title="Text-to-Video Generator", layout="centered")
st.title("🎥 AI Text-to-Video Generator")
st.markdown("Generate short cinematic videos from text using Stable Diffusion and Stable Video Diffusion.")

# Hugging Face Token
hf_token = st.text_input("🔐 Enter your Hugging Face Token", type="password")
if hf_token and "authenticated" not in st.session_state:
    login(hf_token)
    st.session_state.authenticated = True
    st.success("✅ Logged in to Hugging Face!")

# Prompt input
prompt = st.text_input("📝 Enter your prompt:", value="a cinematic shot of a futuristic city with flying cars at sunset")

# Generate button
if st.button("🚀 Generate Video") and hf_token:
    with st.spinner("Loading Text-to-Image model..."):
        text2img_pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        ).to("cuda")

        image = text2img_pipe(prompt, num_inference_steps=50).images[0]
        st.image(image, caption="🖼️ Generated Initial Image", use_column_width=True)

    with st.spinner("Loading Video Generator..."):
        video_pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch.float16,
            variant="fp16"
        ).to("cuda")
        video_pipe.enable_model_cpu_offload()

        video_frames = video_pipe(image, decode_chunk_size=4, generator=torch.manual_seed(42)).frames[0]

        video_path = os.path.join(tempfile.gettempdir(), "generated_video.mp4")
        imageio.mimsave(video_path, video_frames, fps=6)

        st.success("🎉 Video generated successfully!")
        st.video(video_path)
else:
    st.info("Enter your Hugging Face token and a prompt to get started.")
