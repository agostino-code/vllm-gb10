import time
import argparse
from openai import OpenAI

def run_benchmark(prompt, base_url, model_name):
    client = OpenAI(base_url=base_url, api_key="sk-unused")
    
    print(f"--- Inizio benchmark ---")
    print(f"Prompt: {prompt[:50]}...")
    
    start_time = time.time()
    
    # Streaming response to calculate tokens/s accurately
    stream = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    
    first_token_time = None
    token_count = 0
    
    for chunk in stream:
        if first_token_time is None:
            first_token_time = time.time()
        
        if chunk.choices[0].delta.content:
            token_count += 1
            
    end_time = time.time()
    
    total_time = end_time - start_time
    ttft = first_token_time - start_time
    tps = token_count / (end_time - first_token_time)
    
    print(f"--- Risultati ---")
    print(f"Tempo primo token (TTFT): {ttft:.4f} s")
    print(f"Token generati: {token_count}")
    print(f"Tempo totale: {total_time:.4f} s")
    print(f"Velocità (TPS): {tps:.2f} tok/s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Scrivi un breve racconto di fantascienza su una civiltà che vive su un pianeta gassoso.")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", type=str, default="azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ")
    args = parser.parse_args()
    
    run_benchmark(args.prompt, args.url, args.model)
