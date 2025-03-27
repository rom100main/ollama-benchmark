"""
Author: rom100main
Website: http://rom100main.free.fr/
Version: 1.1
"""

from typing import List
import argparse
from datetime import datetime
import json
import subprocess
import os
from pathlib import Path
import platform
import ollama

def get_computer_info():
    try:
        ollama_version_output = subprocess.check_output(['ollama', '--version'], text=True).strip()
        ollama_version = ollama_version_output.split(' ')[-1]
    except:
        ollama_version = "unknown"
        
    return {
        "computer_name": platform.node(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "ollama_version": ollama_version
    }

def benchmark_inference_speed(
    models: List[str],
    prompt: str,
    num_runs: int = 1
):
    computer_info = get_computer_info()
    computer_name = computer_info["computer_name"]
    data_folder = os.path.join('data', 'speed', computer_name)
    os.makedirs(data_folder, exist_ok=True)
    
    info_file = os.path.join(data_folder, 'computer_info.json')
    with open(info_file, 'w') as f:
        json.dump(computer_info, f, indent=2)

    max_name_len = max(len(model) for model in models)
    progress_width = len(f"{num_runs}/{num_runs} runs")

    installed_models = [m.model for m in ollama.list().models]

    for model in models:
        if model not in installed_models:
            print(
                f"\r🚫 "
                f"| {model.ljust(max_name_len)} "
                f"| {f"0/{num_runs} runs".ljust(progress_width)} "
                f"| Model not installed, dont forget to add `:latest` to the model name",
                flush=True
            )
            continue

        total_tps = 0.0
        current_run = 0
        successful_runs = 0
        run_results = []

        print(
            f"\r⏳ "
            f"| {model.ljust(max_name_len)} "
            f"| {f"{current_run}/{num_runs} runs".ljust(progress_width)} "
            f"| Starting...",
            end='', flush=True
        )

        for i in range(num_runs):
            current_run = i + 1
            tps = None

            try:
                response = ollama.generate(
                    model=model,
                    prompt=prompt,
                    stream=False
                )

                if not response.get('done', False):
                    continue

                eval_count = response.get('eval_count')
                eval_duration = response.get('eval_duration')

                if None in (eval_count, eval_duration):
                    continue

                if eval_duration <= 0:
                    continue

                tps = eval_count / (eval_duration / 1e9)
                total_tps += tps
                successful_runs += 1

                run_results.append({
                    'run': current_run,
                    'tokens_per_second': tps,
                    'eval_count': eval_count,
                    'eval_duration': eval_duration
                })

                avg_tps = total_tps / successful_runs

                print(
                    f"\r⏳ "
                    f"| {model.ljust(max_name_len)} "
                    f"| {f"{current_run}/{num_runs} runs".ljust(progress_width)} "
                    f"| Average: {avg_tps:.2f} tokens/sec",
                    end='', flush=True
                )

            except Exception as e:
                run_results.append({
                    'run': current_run,
                    'error': str(e)
                })
                continue

        total_eval_count = sum(run['eval_count'] for run in run_results if 'eval_count' in run)
        result = {
            'timestamp': datetime.now().isoformat(),
            'model': model,
            'prompt': prompt,
            'total_runs': num_runs,
            'successful_runs': successful_runs,
            'average_tokens_per_second': total_tps / successful_runs if successful_runs > 0 else 0,
            'total_eval_count': total_eval_count,
            'runs': run_results
        }

        output_file = os.path.join(data_folder, f"{model.replace("/", "-")}.json")
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

        if successful_runs == 0:
            print(
                f"\r⚠️ "
                f"| {model.ljust(max_name_len)} "
                f"| {f"{successful_runs}/{num_runs} runs".ljust(progress_width)} "
                f"| No successful runs completed",
                flush=True
            )
            continue

        avg_tps = total_tps / successful_runs
        print(
            f"\r✅ "
            f"| {model.ljust(max_name_len)} "
            f"| {f"{successful_runs}/{num_runs} runs".ljust(progress_width)} "
            f"| {f"Average: {avg_tps:.2f} tokens/sec"} ".ljust(len("Average: 999.99 tokens/sec")),
            f"| Total tokens: {total_eval_count}",
            flush=True
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--models',   type=str, nargs='+', help='ollama models to benchmark')
    parser.add_argument('-p', '--prompt',   type=str,            help='prompt to use for generation (default: "Why is the sky blue?")')
    parser.add_argument('-n', '--num_runs', type=int,            help='number of benchmark runs (default: 1)')
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        exit(1)

    benchmark_inference_speed(
        models   = args.models,
        prompt   = args.prompt   if args.prompt   != None else "Why is the sky blue?",
        num_runs = args.num_runs if args.num_runs != None else 1
    )
