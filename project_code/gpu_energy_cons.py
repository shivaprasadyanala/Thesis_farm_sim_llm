import os
import urllib.request

HOST = "http://host.docker.internal:8765"
TOKEN = os.environ["A_LEMS_TOKEN"]


def gpu_collector(action):
    request = urllib.request.Request(
        f"{HOST}/{action}",
        data=b"",
        headers={
            "Authorization": f"Bearer {TOKEN}"
        },
        method="POST"
    )

    with urllib.request.urlopen(request, timeout=10) as response:
        return response.read().decode()


# Start GPU energy measurement on Ubuntu host
print(gpu_collector("start"))

try:
    # --------------------------------
    # Run your A-LEMS / LLM workload
    # --------------------------------
    
    # python agentic_approach_baseline_farm_sim.py --model  Qwen/Qwen3.8-27B --vllm --preset easy --objective profit_hard
    # os.system(" python agentic_approach_baseline_farm_sim.py --model  gemma4:26b --preset easy --objective starter_easy")
    os.system("python run_instruction_catalogue.py --model gemma4:26b --preset normal --tasks harvest_crop_task --repeats 5 --resume")
    # os.system("python run_instruction_catalogue.py --model all --preset normal --tasks harvest_crop_task --repeats 5 --resume")


finally:
    # Stop GPU measurement on Ubuntu host

    print(gpu_collector("stop"))


# Read the GPU measurements
with open("/telemetry/gpu_energy.log", "r") as f:
    print("reading the log")
    print(f.read())
