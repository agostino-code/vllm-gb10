#!/usr/bin/env python3
import torch
import os
import argparse
from safetensors.torch import load_file

def generate_steering_vector(output_path, dimension=2560, mode='random', strength=1.0, source_safetensors=None):
    """
    Genera o converte un vettore di steering salvato come file .pt.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if source_safetensors:
        # Carica da safetensors (es. orca_direction.safetensors)
        data = load_file(source_safetensors)
        # Assumiamo che il vettore sia la prima chiave o 'orca_refusal_direction'
        key = 'orca_refusal_direction' if 'orca_refusal_direction' in data else list(data.keys())[0]
        vec = data[key].to(torch.float32)
        vec = vec / vec.norm() * strength
        print(f"Convertito {source_safetensors} [{key}] in {output_path}")
    elif mode == 'random':
        vec = torch.randn(dimension)
        vec = vec / vec.norm() * strength
        print(f"Vettore casuale generato in {output_path}")
    else:
        vec = torch.zeros(dimension)
        print(f"Vettore nullo generato in {output_path}")
    
    torch.save(vec, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--mode", choices=['random', 'identity', 'convert'], default='random')
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--source", help="Path al file .safetensors da convertire")
    
    args = parser.parse_args()
    if args.mode == 'convert' and not args.source:
        parser.error("--mode convert richiede --source")
        
    generate_steering_vector(args.path, mode=args.mode, strength=args.strength, source_safetensors=args.source)
