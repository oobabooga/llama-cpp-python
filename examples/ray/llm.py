from typing import Dict

from llama_cpp import Llama
from starlette.requests import Request

from ray import serve
from ray.serve import Application


@serve.deployment
class LlamaDeployment:
    def __init__(self, model_path: str):
        self._llm = Llama(model_path=model_path)

    async def __call__(self, http_request: Request) -> Dict:
        input_json = await http_request.json()
        prompt = input_json["prompt"]
        max_tokens = input_json.get("max_tokens", 64)
        return self._llm(prompt, max_tokens=max_tokens)


def llm_builder(args: Dict[str, str]) -> Application:
    return LlamaDeployment.bind(args["model_path"])
