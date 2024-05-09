import asyncio
from ollama import AsyncClient, Client
import aiofiles
import json
import os
import time
from tqdm import tqdm

# Constants and global settings
CACHE_FILE = "models_cache.json"
CACHE_EXPIRY_SECONDS = 86400  # 24 hours in seconds
proxy = None
REQUEST_TIMEOUT = 10
initial_concurrency = 10

async def download_model(ip, model):
    try:
        client = AsyncClient(host=f"{ip}:11434/", proxies=proxy)
        client.pull(model=model)
    except Exception as e:
        print(f"Failed to download model {model} from {ip}: {e}")

async def download_models(model_name):
    if is_cache_valid():
        models_dict = load_cache()
        print("Loaded models from cache")
    else:
        print("Fetching models...")
        models_dict = await fetch_and_cache_models()

    if model_name in models_dict:
        model_ips = models_dict[model_name]
        print(f"Downloading model {model_name} from IPs: {model_ips}")
        tasks = [download_model(ip, model_name) for ip in model_ips]
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Downloading models"):
            await f
    else:
        print(f"Model {model_name} not found.")
# Helper functions
async def fetch_models(ip, models_dict, semaphore):
    async with semaphore:
        try:
            client = AsyncClient(host=f"{ip}:11434/", proxies=proxy)
            model_list = await asyncio.wait_for(client.list(), timeout=REQUEST_TIMEOUT)
            for model in model_list['models']:
                model_name = model['name']
                if model_name in models_dict:
                    models_dict[model_name].add(ip)
                else:
                    models_dict[model_name] = {ip}
        except Exception as e:
            print(f"Failed to fetch models from {ip}: {e}")

async def fetch_and_cache_models():
    models_dict = {}
    semaphore = asyncio.Semaphore(initial_concurrency)
    async with aiofiles.open('ips.txt', 'r') as file:
        content = await file.read()
        ips = set(content.splitlines())

    tasks = [fetch_models(ip, models_dict, semaphore) for ip in ips]
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing IPs"):
        await f

    # Save the cache to a file
    with open(CACHE_FILE, 'w') as cache_file:
        json.dump({"timestamp": time.time(), "data": {k: list(v) for k, v in models_dict.items()}}, cache_file)

    return models_dict

def load_cache():
    with open(CACHE_FILE, 'r') as cache_file:
        cache = json.load(cache_file)
        return cache["data"]

def is_cache_valid():
    if os.path.exists(CACHE_FILE):
        file_timestamp = os.path.getmtime(CACHE_FILE)
        current_time = time.time()
        return (current_time - file_timestamp) < CACHE_EXPIRY_SECONDS
    return False

async def async_check_ip_speed(ip, model):
    try:
        client = AsyncClient(host=f"{ip}:11434/", timeout=60, proxies=proxy)
        result = await client.chat(model=model, messages=[{'role': 'user', 'content': 'Write a long poem about why the sky is blue'}])
        eval_duration = result['eval_duration'] * 1e-9  # Convert from nanoseconds
        tokens_sec = result['eval_count'] / eval_duration
        return (ip, tokens_sec)
    except Exception as e:
        print(f"Error with IP {ip}: {e}")
        return (ip, 0)

async def async_get_fastest_ips(model, model_ips):
    tasks = [async_check_ip_speed(ip, model) for ip in model_ips]
    results = []
    for future in asyncio.as_completed(tasks):
        result = await future
        if result:
            results.append(result)
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# Main function
async def test_model_speed(model_name):
    if is_cache_valid():
        models_dict = load_cache()
        print("Loaded models from cache")
    else:
        print("Fetching models...")
        models_dict = await fetch_and_cache_models()

    if model_name in models_dict:
        model_ips = models_dict[model_name]
        print(f"Testing speeds for model: {model_name} on IPs: {model_ips}")
        fastest_ips = await async_get_fastest_ips(model_name, model_ips)
        print("Fastest IPs in descending order of speed:", fastest_ips)
        return fastest_ips
    else:
        print(f"Model {model_name} not found.")
        return []

def get_fastest_ips(model_name):
    return asyncio.run(test_model_speed(model_name))

def get_openai_url(model_name, vision_model="llava:34b"):
    ips = get_fastest_ips(model_name)
    fastest = ips[0][0]

    client = Client(host=f"{fastest}:11434", proxies=proxy)

    try:
        client.chat(vision_model)
    except client.ResponseError as e:
        print('Error:', e.error)
        if e.status_code == 404:
            client.pull(vision_model)

    return f"http://{fastest}:11434/api/"



# Example usage
if __name__ == "__main__":
    # ips = asyncio.run(test_model_speed("llama3:70b"))
    # print(ips)
    asyncio.run(download_models("llama3:70b"))