from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL
from diffusers.utils import load_image, make_image_grid
from controlnet_aux.pidi import PidiNetDetector
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
import torch
from PIL import Image
import io
import shutil
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

app = FastAPI()

# Slack configuration
slack_token = "{api token}"  # Replace with your actual token
slack_channel = "#art"  # Replace with your actual channel name
slack_client = WebClient(token=slack_token)

# Load adapter
adapter = T2IAdapter.from_pretrained(
    "TencentARC/t2i-adapter-sketch-sdxl-1.0", torch_dtype=torch.float16, variant="fp16"
).to("cuda")

# Load scheduler and VAE
model_id = 'stabilityai/stable-diffusion-xl-base-1.0'
euler_a = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
    model_id, vae=vae, adapter=adapter, scheduler=euler_a, torch_dtype=torch.float16, variant="fp16", 
).to("cuda")
pidinet = PidiNetDetector.from_pretrained("lllyasviel/Annotators").to("cuda")

# BLIP processor and model for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model_blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")

# Sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

import base64

@app.post("/process-image")
async def process_image(file: UploadFile = File(...), prompt: str = Form(...)):
    # Save received image to disk
    image_path = "received_image.png"
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load and process the image with PidiNet
    image = load_image(image_path).convert("RGB")
    image = pidinet(image, detect_resolution=1024, image_resolution=1024, apply_filter=True)

    # Set prompts
    positive_prompt = prompt
    negative_prompt = "extra digit, fewer digits, cropped, worst quality, low quality, glitch, deformed, mutated, ugly, disfigured"

    # Generate image with AI model
    generated_image = pipe(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        image=image,
        num_inference_steps=25,
        adapter_conditioning_scale=0.59,
        guidance_scale=15.0, 
    ).images[0]

    # Caption and sentiment analysis
    raw_image = Image.open(image_path).convert('RGB')
    inputs = processor(generated_image, return_tensors="pt").to("cuda", torch.float16)
    out = model_blip.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    sentiment_result = sentiment_pipeline(caption)

    # Save the generated image to a byte array
    img_byte_arr = io.BytesIO()
    generated_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)  # Reset file pointer to the beginning

    # Encode the image bytes as base64
    encoded_img_bytes = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    # Send the image to Slack
    try:
        slack_client.files_upload(
            channels=slack_channel,
            file=img_byte_arr.getvalue(),
            title="Generated Image",
            initial_comment=f"Caption: {caption}\nSentiment: {sentiment_result}"
        )
    except SlackApiError as e:
        return JSONResponse(status_code=500, content={"message": f"Failed to send image to Slack: {str(e)}"})

    # Send the encoded image along with the JSON response
    response_data = {
        "message": "Image processed and sent to Slack successfully, with caption and sentiment analysis.",
        "image_base64": encoded_img_bytes
    }

    return JSONResponse(content=response_data)

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)