from ollama_calls import get_openai_url

VISION_MODEL = "llava:34b"
MAIN_MODEL = "llama3:70b"

url = get_openai_url(MAIN_MODEL, VISION_MODEL)

def get_url():
    return url, MAIN_MODEL, VISION_MODEL