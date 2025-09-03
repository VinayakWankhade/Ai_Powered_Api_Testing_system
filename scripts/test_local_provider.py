import asyncio
import json
import sys
from pathlib import Path

# Ensure src/ is on PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm_gateway import LLMGateway  # noqa: E402


async def main():
    cfg_path = ROOT / "config" / "gateway.local.json"
    if not cfg_path.exists():
        print(f"Config not found at {cfg_path}. Did you create it?")
        return

    with cfg_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    gateway = LLMGateway(config)

    # Give the gateway a moment to finish async initialization
    await asyncio.sleep(1.0)

    # Use the first configured local model if present
    local_models = config.get("providers", {}).get("local", {}).get("default_models", [])
    model_pref = local_models[0] if local_models else None

    try:
        resp = await gateway.generate_text(
            prompt="Explain the difference between HTTP/1.1 and HTTP/2 in simple terms.",
            model_preference=model_pref,
            max_tokens=512,
            temperature=0.7,
            context={"task_type": "general"}
        )
        print("=== Local Provider Smoke Test ===")
        print(f"Model: {resp.model_used} | Provider: {resp.provider_used.value}")
        print(f"Latency: {resp.latency_ms} ms | Cost: ${resp.cost_usd:.6f}")
        print("Response:\n")
        print(resp.content)
    except Exception as e:
        print("Smoke test failed.")
        print(str(e))
        print("\nTroubleshooting:")
        print("1) Ensure Ollama is installed and running: https://ollama.com/download")
        print("2) Pull a model, e.g.: ollama pull llama3.1:8b")
        print("3) Verify the server is up: Invoke-WebRequest http://localhost:11434/api/tags")


if __name__ == "__main__":
    asyncio.run(main())

