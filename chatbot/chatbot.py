from transformers import pipeline

# Load AI Chatbot using FLAN-T5
chatbot_model = pipeline("text2text-generation", model="google/flan-t5-base")

# Generate Medical Advice
def get_medical_advice(image_description, classification):
    prompt = f"""
    Based on the given lung CT scan description: "{image_description}" and classification result: "{classification}",
    provide a detailed medical diagnosis. If cancerous, suggest the best possible treatments and medicines.
    """

    response = chatbot_model(prompt, max_length=200)
    generated_text = response[0]["generated_text"]

    # Extract Diagnosis & Medicine Information
    diagnosis = generated_text.split("Diagnosis:")[1].split("Medicines:")[0].strip() if "Diagnosis:" in generated_text else "Consult a doctor for accurate results."
    medicines = generated_text.split("Medicines:")[1].strip() if "Medicines:" in generated_text else "No specific medication suggested."

    return diagnosis, medicines
