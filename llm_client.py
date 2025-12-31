from typing import Dict, List
from openai import OpenAI

# Maximum number of conversation turns to include (to limit token usage)
MAX_HISTORY_TURNS = 5

def generate_response(openai_key: str, user_message: str, context: str, 
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """
    Generate response using OpenAI with context.
    
    Args:
        openai_key: OpenAI API key
        user_message: Current user question
        context: Retrieved context from RAG system
        conversation_history: List of previous messages (role + content)
        model: OpenAI model to use
    
    Returns:
        Generated response string
    """
    # Define system prompt positioning assistant as NASA expert who cites sources
    system_prompt = """You are a NASA mission expert. Use the provided context to answer questions accurately and cite your sources from the retrieved documents. If the context is insufficient, state that you cannot answer based on the available information."""
    
    # Build messages list starting with system prompt
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add windowed conversation history (only last N turns to limit tokens)
    # Each "turn" is a user message + assistant response pair
    if conversation_history:
        # Limit to last MAX_HISTORY_TURNS * 2 messages (user + assistant pairs)
        max_messages = MAX_HISTORY_TURNS * 2
        trimmed_history = conversation_history[-max_messages:]
        messages.extend(trimmed_history)
    
    # Build current user message with context included
    # This avoids the extra "ack" message and keeps context with the question
    if context:
        current_message = f"""Based on the following retrieved context, please answer my question.

{context}

Question: {user_message}"""
    else:
        current_message = user_message
    
    messages.append({"role": "user", "content": current_message})
    
    # Create OpenAI Client
    client = OpenAI(api_key=openai_key)
    
    # Send request to OpenAI
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=500
    )
    
    # Return response
    return response.choices[0].message.content