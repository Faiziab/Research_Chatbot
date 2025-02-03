# model_utils.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model():
    """
    Load the DeepSeek R1 1.5B model for question answering with optimizations
    """
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        torch_dtype=torch.float16,  # Use half precision
        device_map="auto"  # Automatically handle device placement
    )
    
    # Enable model evaluation mode
    model.eval()
    
    return model, tokenizer

def generate_answer(question, context, model, tokenizer):
    """
    Generate an answer using the DeepSeek model with optimized settings
    """
    # Construct a more concise prompt
    input_text = f"Context: {context}\nQ: {question}\nA:"
    
    # Tokenize input with optimized settings
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,  # Reduced max length
        padding=True,
        return_attention_mask=True
    )
    
    # Move inputs to the same device as the model
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():  # Disable gradient calculations
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=5000,  # Reduced max length for faster generation
            min_length=30,   # Reduced min length
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            use_cache=True,
            num_beams=2,     # Reduced beam size
        )
    
    # Decode and return the generated answer
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = answer.replace(input_text, "").strip()
    
    return answer

# # model_utils.py
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# # Load a model that's designed for summarization (e.g., BART for summarization)
# def load_model():
#     tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
#     model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
#     return model, tokenizer

# # Summarize the answer using the chosen model
# def generate_answer(question, context, model, tokenizer):
#     input_text = f"Question: {question}\nContext: {context}"
    
#     # Handle long inputs by truncating if necessary
#     inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
    
#     # Generate the summary
#     summary_ids = model.generate(inputs['input_ids'], max_length=200, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    
#     # Decode the generated summary
#     summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#     return summary
