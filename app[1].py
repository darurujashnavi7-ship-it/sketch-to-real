import streamlit as st
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from PIL import Image, ImageOps

st.set_page_config(page_title="Shoe Sketch to Product Image", page_icon="👟", layout="wide")
st.title("👟 Shoe Sketch to Product Image")
st.write("Upload a shoe sketch and generate a studio-style product image guided by its shape.")

CONTROLNET_ID = "lllyasviel/control_v11p_sd15_scribble"
BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"

@st.cache_resource(show_spinner="Loading the image generation models (first run may take several minutes)...")
def load_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_ID, torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        BASE_MODEL_ID,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    if device == "cuda":
        pipe.enable_attention_slicing()
    return pipe, device

uploaded = st.file_uploader("Upload your shoe sketch", type=["png", "jpg", "jpeg", "webp"])

col1, col2 = st.columns(2)
with col1:
    description = st.text_input("Shoe details (optional)", placeholder="e.g., white running shoe with blue accents")
    steps = st.slider("Generation steps", min_value=15, max_value=40, value=25)
    guidance = st.slider("Sketch guidance", min_value=3.0, max_value=12.0, value=7.5, step=0.5)
    seed = st.number_input("Seed (use the same seed to repeat a result)", min_value=0, max_value=2147483647, value=42)

if uploaded:
    sketch = Image.open(uploaded).convert("RGB")
    # Fit the entire sketch onto a square canvas with a margin, avoiding crop/stretch.
    fitted = ImageOps.contain(sketch, (448, 448), method=Image.Resampling.LANCZOS)
    conditioned_sketch = Image.new("RGB", (512, 512), "white")
    conditioned_sketch.paste(fitted, ((512 - fitted.width) // 2, (512 - fitted.height) // 2))
    with col2:
        st.image(sketch, caption="Uploaded sketch", use_container_width=True)

    if st.button("Generate product image", type="primary"):
        prompt = (
            "Photorealistic professional product photograph of a shoe. "
            + (description.strip() + ". " if description.strip() else "")
            + "Preserve the exact overall silhouette, toe shape, heel shape, sole profile, and major design lines. "
            "Show the entire shoe in side view, with clear space around it; do not crop or zoom in. "
            "Realistic materials and stitching, centered single shoe, "
            "commercial studio lighting, clean white background, sharp focus."
        )
        negative_prompt = "multiple shoes, extra parts, deformed shoe, floating object, text, watermark, blurry, low quality"
        try:
            pipe, device = load_pipeline()
            st.caption(f"Running on {device.upper()}. The first generation downloads the models.")
            generator = torch.Generator(device=device).manual_seed(int(seed))
            with st.spinner("Generating image..."):
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=conditioned_sketch,
                    num_inference_steps=int(steps),
                    guidance_scale=float(guidance),
                    controlnet_conditioning_scale=1.15,
                    generator=generator,
                ).images[0]
            st.subheader("Generated product image")
            st.image(result, use_container_width=True)
            import io
            buffer = io.BytesIO()
            result.save(buffer, format="PNG")
            st.download_button("Download image", data=buffer.getvalue(), file_name="generated_shoe.png", mime="image/png")
        except Exception as exc:
            st.error(f"Image generation failed: {exc}")
            st.info("Check that the packages installed successfully and that your computer has enough memory. A CUDA GPU is strongly recommended; CPU generation can be very slow.")
else:
    st.info("Choose a clear sketch with the shoe centered and visible. High contrast line drawings usually work best.")

st.caption("This app uses pretrained Stable Diffusion + ControlNet models from your notebook. The notebook does not train a custom shoe model, so results can vary and may not reproduce every sketch detail exactly.")

