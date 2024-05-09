import asyncio

from ollama import Client
from ollama_calls import get_fastest_ips
from openai import OpenAI

using_ollama = True

if using_ollama:
    MODEL_NAME = "llama3:70b"
    ips = get_fastest_ips(MODEL_NAME)
    if ips:
        fastest_ip = ips[0][0]
        print(f"Using fastest IP: {fastest_ip}")
        ollama_client = Client(host=f"{fastest_ip}:11434/", timeout=120)

else:
    client = OpenAI(api_key="insert_your_api_key_here")
    # Available models: "gpt-4-1106-preview", "gpt-3.5-turbo-1106", or "davinci-codex"
    MODEL_NAME = "gpt-3.5-turbo-1106"





def api_call(messages, model_name=MODEL_NAME, temperature=0.5, max_tokens=150):
    if model_name == "gpt-4-1106-preview":
        model_name = "llama3:70b"
    try:
        if using_ollama:
            response = ollama_client.chat(model=model_name, messages=messages, temperature=temperature, max_tokens=max_tokens)
            decision = response.choices[0].message.content.strip()
        else:
        # Execute the chat completion using the chosen model
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                # Additional configurations can be passed as parameters here
                temperature=temperature,  # Values can range from 0.0 to 1.0
                max_tokens=max_tokens,  # This specifies the maximum length of the response
                # Tip: adding more configurations as needed
            )

        # Since we're not using 'with_raw_response', 'response' is now the completion object
        if response.choices and hasattr(response.choices[0], "message"):
            decision_message = response.choices[0].message

            # Make sure we have 'content' in the message
            if hasattr(decision_message, "content"):
                decision = decision_message.content.strip()
            else:
                decision = None
        else:
            decision = None

        return decision
    except Exception as e:
        raise Exception(f"An error occurred: {e}")


# # Replace this payload with the actual messages sequence for your use case # # Test
# messages_payload = [
#     {"role": "system", "content": "You are a helpful and knowledgeable assistant. Always uwufy the text."},
#     {"role": "user", "content": "Please help me troubleshoot my JavaScript code."}
# ]
#
# # Example configuration: you might want to specify 'temperature' for more creative responses,
# # or 'max_tokens' for more concise outputs
# result = api_call(messages_payload, temperature=0.7, max_tokens=100)
# print(f"AI Analysis Result: '{result}'")
