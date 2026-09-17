import json
from typing import Any, Dict, List


class NarrativePromptBuilder:
    LABEL_TO_PT = {
        # Pessoas
        "person": "pessoa",
        "woman": "mulher",
        "man": "homem",
        "child": "criança",

        # Aparência
        "long_hair": "cabelos longos",
        "short_hair": "cabelos curtos",
        "black_clothes": "roupas pretas",
        "white_clothes": "roupas brancas",
        "gray_clothes": "roupas cinzas",
        "brown_clothes": "roupas marrons",
        "dress": "vestido",
        "glasses": "óculos",

        # Objetos
        "umbrella": "guarda-chuva",
        "backpack": "mochila",
        "bag": "bolsa",
        "handbag": "bolsa de mão",
        "bicycle": "bicicleta",
        "window": "janela",

        # Ações
        "walking": "caminhando",
        "running": "correndo",
        "standing": "em pé",
        "sitting": "sentado ou sentada",
        "moving": "em movimento",
        "still": "parado",
        "playing": "brincando",
        "working": "trabalhando",

        # Cenário
        "outdoor": "ao ar livre",
        "indoor": "em ambiente interno",
        "street": "rua",
        "road": "estrada",
        "city": "cidade",
        "park": "parque",
        "field": "campo",
        "ocean": "oceano",

        # Atmosfera
        "sunny": "ensolarado",
        "cloudy": "nublado",
        "rainy": "chuvoso",
        "foggy": "com neblina",
        "snowy": "com neve",
        "clear_weather": "tempo limpo",
        "low_light": "baixa iluminação",
        "bright": "bem iluminado",
        "day": "durante o dia",
        "night": "durante a noite",
        "dawn_dusk": "ao amanhecer ou entardecer",
    }

    def translate_values(
        self,
        values: List[str],
    ) -> List[str]:
        return [
            self.LABEL_TO_PT.get(value, value)
            for value in values
        ]

    def build(
        self,
        data: Dict[str, Any],
    ) -> str:
        translated_data = {}

        for key, value in data.items():
            if isinstance(value, list):
                translated_data[key] = self.translate_values(
                    value
                )
            else:
                translated_data[key] = value

        input_json = json.dumps(
            translated_data,
            ensure_ascii=False,
            indent=2,
        )

        return f"""
            Converta os FATOS em UMA frase natural de audiodescrição.

            REGRAS:
            - Use somente os FATOS fornecidos.
            - Não invente nenhuma informação.
            - Não escreva uma lista.
            - Não enumere palavras separadas por vírgulas.
            - Construa uma frase gramatical completa.
            - Comece pelo sujeito quando houver sujeito.
            - Em seguida descreva a ação ou postura.
            - Depois acrescente objeto, aparência, local e atmosfera quando existirem.
            - Não repita informações equivalentes.
            - Se houver "mulher", não diga também "pessoa".
            - Se houver "homem", não diga também "pessoa".
            - Não crie ação quando "action" estiver vazio.
            - Escreva em português natural.
            - Máximo de 25 palavras.
            - Retorne somente a frase.

            FATOS:
            {input_json}

            FRASE:
            """.strip()