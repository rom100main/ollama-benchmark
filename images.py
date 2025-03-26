"""
Author: rom100main
Website: http://rom100main.free.fr/
Version: 1.0
"""

from typing import List, Union, Tuple
from datetime import datetime
import json
import base64
import argparse
import os
from pathlib import Path
import ollama

def is_valid_image(image_path: str) -> Tuple[bool, str]:
    try:
        with open(image_path, 'rb') as f:
            header = f.read(8)
            if header.startswith(b'\x89PNG\r\n\x1a\n'): return True, "PNG"
            if header.startswith(b'\xFF\xD8\xFF'): return True, "JPEG"
            return False, "Not a PNG/JPEG image"
    except Exception as e:
        return False, f"Error reading file: {str(e)}"

def get_images_from_folder(folder_path: str) -> List[str]:
    valid_extensions = {'.png', '.jpg', '.jpeg'}
    images = []
    
    try:
        for entry in Path(folder_path).rglob('*'):
            if entry.is_file() and entry.suffix.lower() in valid_extensions:
                images.append(str(entry))
    except Exception as e:
        print(f"Error scanning folder {folder_path}: {str(e)}")
        return []
        
    return sorted(images)

def encode_image(image_path: str) -> str:
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_prompt_from_file(image_path: str) -> str:
    prompt_file = f"{os.path.splitext(image_path)[0]}_prompt.md"
    if os.path.exists(prompt_file):
        try:
            with open(prompt_file, 'r') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Warning: Failed to read prompt file {prompt_file}: {e}")
    return None

def process_images(
    models: List[str],
    images: List[str],
    prompt: str = None
):
    default_prompt = """
Convert the image to markdown

Formulae should be in latex format between $ like $a=\\frac{b}{c}$.
Use simple $ not $$.

Output only the markdown text.
"""

    max_name_len = max(len(model) for model in models)
    installed_models = [m["model"] for m in ollama.list()["models"]]

    for image_path in images:
        if not os.path.exists(image_path):
            print(
                f"\r🚫 "
                f"| {os.path.basename(image_path).ljust(max_name_len)} "
                f"| Image not found",
                flush=True
            )
            continue

        is_valid, img_type = is_valid_image(image_path)
        if not is_valid:
            print(
                f"\r🚫 "
                f"| {os.path.basename(image_path).ljust(max_name_len)} "
                f"| Invalid image type: {img_type}",
                flush=True
            )
            continue

        current_prompt = prompt or get_prompt_from_file(image_path) or default_prompt

        image_name = os.path.splitext(os.path.basename(image_path))[0]
        data_folder = os.path.join('data/images/', image_name)
        os.makedirs(data_folder, exist_ok=True)

        encoded_image = encode_image(image_path)

        for model in models:
            if model not in installed_models:
                print(
                    f"\r🚫 "
                    f"| {model.ljust(max_name_len)} "
                    f"| Model not installed, dont forget to add `:latest` to the model name",
                    flush=True
                )
                continue

            print(
                f"\r⏳ "
                f"| {model.ljust(max_name_len)} "
                f"| Processing {image_name}...",
                end='', flush=True
            )

            output_file = os.path.join(data_folder, f"{model}.json")
            
            try:
                response = ollama.chat(
                    model=model,
                    messages=[{
                        'role': 'user',
                        'content': current_prompt,
                        'images': [encoded_image]
                    }],
                    stream=False
                )
                
                result = {
                    'timestamp': datetime.now().isoformat(),
                    'model': model,
                    'image': image_name,
                    'image_type': img_type,
                    'response': response["message"]["content"],
                    'metadata': {
                        'total_duration': response["total_duration"],
                        'load_duration': response["load_duration"],
                        'prompt_eval_duration': response["eval_duration"],
                        'eval_count': response["eval_count"],
                    }
                }
                
                with open(output_file, 'w') as f: json.dump(result, f, indent=2)
                
                print(
                    f"\r✅ "
                    f"| {model.ljust(max_name_len)} "
                    f"| Save {image_name} result to: {output_file}",
                    flush=True
                )
                    
            except Exception as e:
                print(
                    f"\r⚠️ "
                    f"| {model.ljust(max_name_len)} "
                    f"| Error while processing {image_name}: {str(e)}",
                    flush=True
                )
                error_result = {
                    'timestamp': datetime.now().isoformat(),
                    'model': model,
                    'image': image_name,
                    'image_type': img_type,
                    'error': str(e)
                }
                with open(output_file, 'w') as f: json.dump(error_result, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--models', type=str, nargs='+', required=True, help='ollama models to use for image processing')
    parser.add_argument('-i', '--images', type=str, nargs='+', required=True, help='paths to images or folders to process')
    parser.add_argument('-p', '--prompt', type=str, help='custom prompt to use for image processing (overrides prompt files)')
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        exit(1)

    os.makedirs('data', exist_ok=True)
    
    all_images = []
    for path in args.images:
        if os.path.isdir(path):
            all_images.extend(get_images_from_folder(path))
        else:
            all_images.append(path)
            
    if not all_images:
        print("No images found to process")
        exit(1)
    
    process_images(
        models=args.models,
        images=all_images,
        prompt=args.prompt
    )
