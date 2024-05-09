# from core.ollama_calls import get_openai_url

VISION_MODEL = "llava:34b"
MAIN_MODEL = "llama3:70b"

# url = get_openai_url(MAIN_MODEL, VISION_MODEL)
url = "http://74.82.31.213:11434/v1/"

def get_url():
    return url, MAIN_MODEL, VISION_MODEL