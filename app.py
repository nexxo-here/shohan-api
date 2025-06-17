from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionXLPipeline
from safetensors.torch import load_file
import torch
import uuid
import os

app = FastAPI()

# Load base model
model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    revision="fp16",
    safety_checker=None
).to("cuda")

# Load LoRA
lora_path = "./lora/shohan.safetensors"
pipe.load_lora_weights(lora_path)
pipe.fuse_lora()

@app.get("/gen")
async def generate_image(prompt: str = Query(...)):
    image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    filename = f"output_{uuid.uuid4().hex}.png"
    image.save(filename)
    return FileResponse(filename, media_type="image/png")
  
