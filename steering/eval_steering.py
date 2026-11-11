#!/usr/bin/env python3
"""
Valuta l'effetto dello steering vector interrogando il server vLLM attivo.
Esegue un set di prompt di controllo e misura:
- Lunghezza della risposta (token/parole)
- Tempo e velocità di generazione
- Confronto con o senza steering
"""
import argparse
import json
import os
import sys
import time
import urllib.request

DEFAULT_PROMPTS = [
    "Explain why databases use indexes.",
    "How should I learn sorting algorithms?",
    "Describe how clouds form.",
    "Give me advice for debugging a slow program.",
    "What is public key cryptography?",
]

def load_prompts(file_path):
    if not file_path or not os.path.exists(file_path):
        return DEFAULT_PROMPTS
    prompts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.append(line)
    return prompts

def query_server(url, model, prompt, max_tokens=256, temperature=0.0):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": False}
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            elapsed = time.time() - t0
            res = json.loads(resp.read().decode("utf-8"))
            choice = res["choices"][0]["message"]
            content = choice.get("content") or choice.get("reasoning") or ""
            usage = res.get("usage", {})
            gen_tokens = usage.get("completion_tokens", len(content.split()))
            return {
                "content": content.strip(),
                "tokens": gen_tokens,
                "elapsed": elapsed,
                "tok_s": gen_tokens / elapsed if elapsed > 0 else 0
            }
    except Exception as e:
        return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Valutatore Steering per Qwen3.8-Flash")
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--models-url", default="http://127.0.0.1:8000/v1/models")
    parser.add_argument("--prompts", help="Percorso a un file di prompt (es. ds4/dir-steering/examples/eval_prompts.txt)")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--out", help="File JSON per salvare i risultati")
    args = parser.parse_args()

    # Rileva modello attivo
    try:
        req = urllib.request.Request(args.models_url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            m_data = json.loads(resp.read().decode("utf-8"))
            model = m_data["data"][0]["id"]
    except Exception as e:
        print(f"Errore nel rilevare il modello da {args.models_url}: {e}")
        sys.exit(1)

    prompts = load_prompts(args.prompts)
    print(f"==> Modello: {model}")
    print(f"==> Numero prompt: {len(prompts)}")
    print(f"==> Steering Path (env): {os.environ.get('STEERING_VECTOR_PATH', '(nessuno - modello base)')}")
    print(f"==> Steering Strength (env): {os.environ.get('STEERING_STRENGTH', '1.0')}")
    print("-" * 60)

    results = []
    tot_tokens = 0
    tot_time = 0.0

    for idx, p in enumerate(prompts, 1):
        print(f"\n[{idx}/{len(prompts)}] Prompt: {p}")
        res = query_server(args.url, model, p, max_tokens=args.max_tokens)
        if "error" in res:
            print(f"  ERRORE: {res['error']}")
            continue
        tot_tokens += res["tokens"]
        tot_time += res["elapsed"]
        print(f"  -> Generati: {res['tokens']} tokens in {res['elapsed']:.2f}s ({res['tok_s']:.1f} tok/s)")
        print(f"  -> Risposta (anteprima): {res['content'][:120]}...")
        results.append({"prompt": p, "result": res})

    avg_len = tot_tokens / len(results) if results else 0
    avg_speed = tot_tokens / tot_time if tot_time > 0 else 0
    print("\n" + "=" * 60)
    print(f"Riepilogo:")
    print(f"  Token totali: {tot_tokens}")
    print(f"  Lunghezza media risposta: {avg_len:.1f} tokens")
    print(f"  Velocità media: {avg_speed:.1f} tok/s")
    print("=" * 60)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({
                "model": model,
                "steering_path": os.environ.get("STEERING_VECTOR_PATH"),
                "steering_strength": os.environ.get("STEERING_STRENGTH"),
                "summary": {"total_tokens": tot_tokens, "avg_length": avg_len, "avg_speed": avg_speed},
                "details": results
            }, f, indent=2)
        print(f"Risultati salvati in: {args.out}")

if __name__ == "__main__":
    main()
