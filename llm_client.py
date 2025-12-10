from typing import Dict, List
from openai import OpenAI

def generate_response(openai_key: str, user_message: str, context: str, 
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""

    # Define system prompt
    system_prompt = "You are a NASA mission expert. Use the provided context to answer questions accurately and cite your sources from the retrieved documents. If the context is insufficient, state that you cannot answer based on the available information."
    
    # Build messages list
    messages = []
    
    # Set context in messages
    if context:
        messages.append({"role": "user", "content": f"Context: {context}"})
        messages.append({"role": "assistant", "content": "I understand. I'll use this context to help answer your questions."})
    
    # Add chat history
    for msg in conversation_history:
        messages.append(msg)
    
    # Add current user message
    messages.append({"role": "user", "content": user_message})
    
    # Create OpenAI Client
    client = OpenAI(api_key=openai_key)
    
    # Send request to OpenAI
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}] + messages,
        temperature=0.7,
        max_tokens=500
    )
    
    # Return response
    return response.choices[0].message.content