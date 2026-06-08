import json
from typing import Any, Dict


class NarrativePromptBuilder:
    """
    Monta o prompt enviado ao modelo local.

    A ideia é usar o LLM como reescritor narrativo controlado,
    não como inventador livre de cenas.
    """

    def build(self, data: Dict[str, Any]) -> str:
        input_json = json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        )

        return """
Transforme os dados visuais abaixo em uma audiodescrição curta e natural.

Regras obrigatórias:
- Escreva em português do Brasil.
- Use apenas as informações presentes nos dados visuais.
- Não invente idade, gênero, emoção, cor, clima, intenção ou quantidade.
- Não invente ações que não aparecem nos dados.
- Não use termos técnicos como "label", "frame", "modelo", "sistema", "detecção" ou "IA".
- Não diga "a imagem mostra" ou "a cena mostra" se houver sujeito e ação claros.
- Evite repetir a descrição anterior.
- A frase deve ser boa para ser lida em voz alta.
- A frase deve ter no máximo 25 palavras.
- Retorne apenas uma frase final.
- Não use aspas.
- Não use lista.
- Não explique o raciocínio.

Dados visuais:
{}

Audiodescrição:
""".format(input_json).strip()
