import os
from datetime import datetime

from PIL import Image

from lumina_mgpt.inference_solver import FlexARInferenceSolver

# ******************** Image Generation ********************
inference_solver = FlexARInferenceSolver(
    model_path="Alpha-VLLM/Lumina-mGPT-7B-768",
    precision="bf16",
    target_size=768,
)

q1 = f"Generate an image of 768x768 according to the following prompt:\nImage of a dog playing water, and a waterfall is in the background."

# generated: tuple of (generated response, list of generated images)
generated = inference_solver.generate(
    images=[],
    qas=[[q1, None]],
    max_gen_len=8192,
    temperature=1.0,
    logits_processor=inference_solver.create_logits_processor(
        cfg=4.0, image_top_k=2000
    ),
)
# 이미지 생성 마지막 부분
"""
 Allocated memory      |  17287 MiB |  17324 MiB |   5663 GiB |   5646 GiB |
|       from large pool |  17256 MiB |  17293 MiB |   5553 GiB |   5536 GiB |
|       from small pool |     31 MiB |    105 MiB |    109 GiB |    109 GiB |
|---------------------------------------------------------------------------|
| Active memory         |  17287 MiB |  17324 MiB |   5663 GiB |   5646 GiB |
|       from large pool |  17256 MiB |  17293 MiB |   5553 GiB |   5536 GiB |
|       from small pool |     31 MiB |    105 MiB |    109 GiB |    109 GiB |
|---------------------------------------------------------------------------|
| Requested memory      |  17264 MiB |  17300 MiB |   5611 GiB |   5594 GiB |
|       from large pool |  17232 MiB |  17269 MiB |   5502 GiB |   5485 GiB |
|       from small pool |     31 MiB |    105 MiB |    108 GiB |    108 GiB 
"""

a1, new_image = generated[0], generated[1][0]

now = datetime.now()
save_path = f"output_Alpha-VLLM/Lumina-mGPT-7B-768_output_{a1}.png"

if not os.path.exists(save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

new_image.save(save_path)


# ******************* Image Understanding ******************
inference_solver = FlexARInferenceSolver(
    model_path="Alpha-VLLM/Lumina-mGPT-7B-512",
    precision="bf16",
    target_size=512,
)

# "<|image|>" symbol will be replaced with sequence of image tokens before fed to LLM
q1 = "Describe the image in detail. <|image|>"

images = [Image.open(save_path)]
qas = [[q1, None]]

# `len(images)` should be equal to the number of appearance of "<|image|>" in qas
generated = inference_solver.generate(
    images=images,
    qas=qas,
    max_gen_len=8192,
    temperature=1.0,
    logits_processor=inference_solver.create_logits_processor(
        cfg=4.0, image_top_k=2000
    ),
)

a1 = generated[0]

print(a1)
# save the generated text to a file
with open(f"{save_path}_description.txt", "w") as f:
    f.write(a1)


# generated[1], namely the list of newly generated images, should typically be empty in this case.


# ********************* Omni-Potent *********************
inference_solver = FlexARInferenceSolver(
    model_path="Alpha-VLLM/Lumina-mGPT-7B-768-Omni",
    precision="bf16",
    target_size=768,
)

# Example: Depth Estimation
# For more instructions, see demos/demo_image2image.py
q1 = "Depth estimation. <|image|>"
images = [Image.open(save_path)]
qas = [[q1, None]]

generated = inference_solver.generate(
    images=images,
    qas=qas,
    max_gen_len=8192,
    temperature=1.0,
    logits_processor=inference_solver.create_logits_processor(cfg=1.0, image_top_k=200),
)

a1 = generated[0]
new_image = generated[1][0]


save_path = f"output_Alpha-VLLM/Lumina-mGPT-7B-768-Omni_{a1}.png"
new_image.save(save_path)
