"""
Author: rom100main
Website: http://rom100main.free.fr/
Version: 1.0
"""

from typing import List
import argparse
import ollama

def benchmark_inference_speed(
    models: List[str],
    prompt: str,
    num_runs: int = 1
):
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

                avg_tps = total_tps / successful_runs

                print(
                    f"\r⏳ "
                    f"| {model.ljust(max_name_len)} "
                    f"| {f"{current_run}/{num_runs} runs".ljust(progress_width)} "
                    f"| Average: {avg_tps:.2f} tokens/sec",
                    end='', flush=True
                )

            except Exception as e:
                continue

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
            f"| Average: {avg_tps:.2f} tokens/sec",
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
