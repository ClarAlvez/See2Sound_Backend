
from pathlib import Path
from typing import Optional

from llama_cpp import Llama


class LlamaCppClient:
    def __init__(
        self,
        model_path: str = "data/models/qwen/qwen2.5-3b-instruct-q4_k_m.gguf",
        n_ctx: int = 2048,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0,
        temperature: float = 0.15,
        top_p: float = 0.85,
        max_tokens: int = 100,
        verbose: bool = False,
    ):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.verbose = verbose

        self._validate_model_path()

        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_gpu_layers=self.n_gpu_layers,
            chat_format="chatml",
            verbose=self.verbose,
        )

    def generate(self, prompt: str) -> str:
        if not prompt or not prompt.strip():
            return ""

        messages = [
            {
                "role": "system",
                "content": (
                    "Você é um componente de verbalização "
                    "visual do See2Sound. "
                    "Sua função é transformar fatos visuais "
                    "confirmados em uma frase natural de "
                    "audiodescrição, em português brasileiro. "
                    "Não deduza acontecimentos, intenções, "
                    "emoções, posições ou relações entre "
                    "objetos que não estejam explicitamente "
                    "presentes na entrada. "
                    "Use artigos, preposições, verbos de "
                    "ligação e flexões gramaticais para "
                    "formar frases completas. "
                    "Não reproduza listas de palavras. "
                    "Não escreva explicações. "
                    "Retorne somente a audiodescrição."
                ),
            },
            {
                "role": "user",
                "content": prompt.strip(),
            },
        ]

        try:
            response = self.llm.create_chat_completion(
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                repeat_penalty=1.08,
            )

            return self._extract_response_text(response)

        except Exception as error:
            raise RuntimeError(
                "Erro ao gerar audiodescrição com GGUF: "
                f"{error}"
            ) from error

    def _extract_response_text(self, response: dict) -> str:
        try:
            content = response["choices"][0]["message"]["content"]

            if content is None:
                return ""

            return str(content).strip()

        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(
                f"Resposta inesperada do modelo: {response}"
            ) from error

    def _validate_model_path(self) -> None:
        path = Path(self.model_path)

        if not path.is_file():
            raise FileNotFoundError(
                "Modelo GGUF não encontrado: "
                f"{path.resolve()}"
            )

        if path.suffix.lower() != ".gguf":
            raise ValueError(
                f"Formato de modelo inválido: {path}"
            )
