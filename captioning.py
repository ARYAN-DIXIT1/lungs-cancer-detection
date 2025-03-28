import torch
import logging
from transformers import AutoProcessor, AutoModelForCausalLM

# Load model and processor
def load_captioning_model(model_id):
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval().cuda()
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        return model, processor
    except Exception as e:
        logging.error(f"Error loading captioning model: {str(e)}")
        return None, None

def generate_captions(frames, timestamps, model, processor, task_prompt="<MORE_DETAILED_CAPTION>"):
    """Generate captions for given frames."""
    try:
        captions = []
        for i, image in enumerate(frames):
            inputs = processor(text=task_prompt, images=image, return_tensors="pt").to("cuda")
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    do_sample=False,
                    num_beams=3,
                )
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            parsed_text = processor.post_process_generation(generated_text, task=task_prompt)
            captions.append(f"Time {timestamps[i]:.1f}s: {parsed_text}")

        logging.info(f"Generated {len(captions)} captions")
        return captions

    except Exception as e:
        logging.error(f"Error in generate_captions: {str(e)}")
        return []
