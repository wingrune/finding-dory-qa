import os
import tqdm
import re
import ast
import json
from PIL import Image
import torch
import pandas as pd
from transformers import AutoProcessor, AutoModelForVision2Seq

# ---- CONFIGURATION ----
df = pd.read_parquet("../findingdory-subsampled-96/data/validation-00000-of-00001.parquet", engine="fastparquet")
model_name = "../Qwen2.5-VL-7B-Instruct"  # HF model

# ---- LOAD MODEL ----
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForVision2Seq.from_pretrained(model_name).to(device)

answers = []

for idx, row in tqdm.tqdm(df.iterrows()):

    if row["ep_id"] not in ["ep_2", "ep_5", "ep_6", "ep_9"]:
        continue

    content_list = []

    system_prompt = """
        You are an expert and intelligent question answering agent. You will be given a text description of a video that was collected by a robot yesterday while navigating around a house and picking and placing objects.
        Your job is to help the robot complete a task today by looking at the video and selecting the frame indices that the robot should move to.
        Make sure your response is all within the JSON.
        Note:
        - The robot uses a magic grasp action to pick up an object, where a gripper goes close to the object and the object gets magically picked up.
        The description of the video.
        \n
    """

    content_list.append({"type": "text", "text": system_prompt})

    with open(f"captions_16/{row['ep_id']}.json", "r") as f:
        captions = json.load(f)

    frames_dir = f"./subsampled_frames/{row['ep_id']}"                    # folder with image frames
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".png"))])
    frames = []

    for frame_idx, caption in captions.items():
        frame_idx_int = int(re.search(r"(\d+)", frame_idx).group(1))
        content_list.append({"type": "text", "text": f"[Frame {frame_idx_int}] {caption.strip()}. The image:"})
        img_path = os.path.join(frames_dir, f"frame_{frame_idx}.jpg")
        image = Image.open(img_path).convert("RGB")
        image = image.crop((0, 80, 360, 640))
        image = image.resize((180, 280), Image.Resampling.LANCZOS)
        frames.append(image)
        content_list.append({"type": "image", "image": image})
        


    system_prompt = (
        "\nHere is a question about the video:\n" +
        f"Question: {row['question']}\n"
    )

    system_prompt += """
    Use the following format for the answer:
        {
            "frame_indices": [[frame1, frame2, frame3]] selected the frame indices that the robot should move to,
            "explanation": ["Explanation"] detailed explanation why you selected the objects present in these frames.
        }
    """

    content_list.append({"type": "text", "text": system_prompt})

    # --- Build instruction with image tokens ---
    messages = [
        {
            "role": "user",
            "content": content_list
        }
    ]

    # Process inputs
    text_template = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = processor(
        text=[text_template],
        images=frames,
        padding=True,
        return_tensors="pt"
    )
    # 
    inputs = inputs.to(device)

    # Generate caption
    output_ids = model.generate(**inputs, max_new_tokens=1024)
    caption = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    caption = caption.strip().split("assistant")[-1]

    with open(f"answers_time_vl_image_16/{idx}.json", "w") as f:
        try:
            caption = json.loads(caption.split("json")[-1].split("```")[0])
            caption['answer'] = ast.literal_eval(row['answer'])
            caption['question'] = row['question']
            caption['ep_id'] = row['ep_id']
            json.dump(caption, f, indent=2)
        except:
            print(caption)
            pass