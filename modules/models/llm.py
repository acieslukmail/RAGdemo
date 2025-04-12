import openai
import requests
import json

def generate_answer_openai(prompt, model_name, api_key, temperature=0.2):
    """
    Generate answer using OpenAI API.
    
    Args:
        prompt: Text prompt for the model
        model_name: Name of the OpenAI model to use
        api_key: OpenAI API key
        temperature: Temperature parameter for text generation
        
    Returns:
        Generated answer text
    """
    openai.api_key = api_key
    
    response = openai.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a document expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=500
    )
    
    return response.choices[0].message.content

def generate_answer_deepseek(prompt, model_name, api_key, temperature=0.2):
    """
    Generate answer using DeepSeek API.
    
    Args:
        prompt: Text prompt for the model
        model_name: Name of the DeepSeek model to use
        api_key: DeepSeek API key
        temperature: Temperature parameter for text generation
        
    Returns:
        Generated answer text
    """
    url = "https://api.deepseek.com/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a document expert."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 500
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    result = response.json()
    
    if "choices" in result and len(result["choices"]) > 0:
        return result["choices"][0]["message"]["content"]
    else:
        error_msg = result.get("error", {}).get("message", "Unknown API error")
        raise Exception(f"DeepSeek API Error: {error_msg}")

def generate_answer(prompt, provider, model_name, openai_key, deepseek_key, temperature=0.2):
    """
    Generate answer using the specified provider and model.
    
    Args:
        prompt: Text prompt for the model
        provider: Provider name ('OpenAI' or 'DeepSeek')
        model_name: Name of the model to use
        openai_key: OpenAI API key
        deepseek_key: DeepSeek API key
        temperature: Temperature parameter for text generation
        
    Returns:
        Generated answer text
    """
    if provider == "OpenAI":
        return generate_answer_openai(prompt, model_name, openai_key, temperature)
    elif provider == "DeepSeek":
        return generate_answer_deepseek(prompt, model_name, deepseek_key, temperature)
    else:
        raise ValueError(f"Unknown provider: {provider}")