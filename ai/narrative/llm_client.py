from pathlib import Path
from typing import Optional

from llama_cpp import Llama


class LlamaCppClient:
    """
    Cliente local para geração de texto usando modelo GGUF via llama-cpp-python.

    Exemplo de modelo:
    models/Llama-3.2-1B-Instruct-Q6_K_L.gguf
    """

    def __init__(
        self,
        model_path: str = "models/Llama-3.2-1B-Instruct-Q6_K_L.gguf",
        n_ctx: int = 4096,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0,
        temperature: float = 0.55,
        top_p: float = 0.9,
        max_tokens: int = 256,
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
            verbose=self.verbose,
        )

    def generate(self, prompt: str) -> str:
        if not prompt or not prompt.strip():
            return ""

        try:
            response = self.llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Você é o verbalizador visual do sistema See2Sound. "
                            "Sua tarefa é converter dados produzidos por modelos de visão computacional "
                            "em uma frase natural de audiodescrição. "
                            "Você NÃO deve imaginar a cena. "
                            "Você NÃO deve completar informações ausentes. "
                            "Você deve preservar gênero, ação, roupas, acessórios, ambiente e movimento "
                            "quando essas informações estiverem disponíveis. "
                            "Uma ação detectada nunca pode ser substituída por outra. "
                            "Responda sempre em português do Brasil e apenas com a frase final."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt.strip(),
                    },
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )

            return self._extract_response_text(response)

        except Exception as error:
            raise RuntimeError(
                "Erro ao gerar texto com o modelo local GGUF: {}".format(error)
            )

    def _extract_response_text(self, response: dict) -> str:
        try:
            return response["choices"][0]["message"]["content"].strip()
        except Exception:
            raise RuntimeError(
                "Resposta inesperada do modelo local: {}".format(response)
            )

    def _validate_model_path(self) -> None:
        path = Path(self.model_path)

        if not path.exists():
            raise FileNotFoundError(
                "Modelo GGUF não encontrado.\n"
                "Caminho esperado: {}\n\n"
                "Verifique se o arquivo foi baixado e colocado dentro da pasta models/."
                .format(path.resolve())
            )

        if path.suffix.lower() != ".gguf":
            raise ValueError(
                "O arquivo informado não parece ser um modelo GGUF: {}".format(path)
            )
