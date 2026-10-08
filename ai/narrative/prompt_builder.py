
import json
from typing import Any, Dict, List


class NarrativePromptBuilder:
    LABEL_TO_PT = {
        "person": "pessoa",
        "woman": "mulher",
        "man": "homem",
        "child": "criança",
        "boy": "menino",
        "girl": "menina",
        "people": "pessoas",
        "dog": "cachorro",
        "cat": "gato",
        "animal": "animal",

        "long_hair": "cabelos longos",
        "short_hair": "cabelos curtos",
        "black_clothes": "roupas pretas",
        "white_clothes": "roupas brancas",
        "gray_clothes": "roupas cinzas",
        "brown_clothes": "roupas marrons",
        "red_clothes": "roupas vermelhas",
        "blue_clothes": "roupas azuis",
        "green_clothes": "roupas verdes",
        "yellow_clothes": "roupas amarelas",
        "dress": "vestido",
        "glasses": "óculos",

        "umbrella": "guarda-chuva",
        "backpack": "mochila",
        "bag": "bolsa",
        "handbag": "bolsa de mão",
        "bicycle": "bicicleta",
        "bike": "bicicleta",
        "window": "janela",
        "car": "carro",
        "bus": "ônibus",
        "train": "trem",
        "boat": "barco",
        "phone": "telefone",
        "book": "livro",
        "table": "mesa",
        "chair": "cadeira",
        "tree": "árvore",
        "computer": "computador",
        "door": "porta",
        "ball": "bola",

        "walking": "caminhando",
        "running": "correndo",
        "standing": "em pé",
        "sitting": "sentado",
        "moving": "em movimento",
        "still": "parado",
        "playing": "brincando",
        "working": "trabalhando",
        "talking": "falando",
        "looking": "olhando",
        "jumping": "saltando",
        "dancing": "dançando",
        "eating": "comendo",
        "drinking": "bebendo",
        "sleeping": "dormindo",
        "reading": "lendo",
        "writing": "escrevendo",
        "swimming": "nadando",
        "holding": "segurando",

        "outdoor": "ao ar livre",
        "indoor": "em ambiente interno",
        "street": "rua",
        "road": "estrada",
        "city": "cidade",
        "park": "parque",
        "field": "campo",
        "ocean": "oceano",
        "forest": "floresta",
        "beach": "praia",
        "kitchen": "cozinha",
        "bedroom": "quarto",
        "living_room": "sala de estar",
        "office": "escritório",
        "restaurant": "restaurante",
        "school": "escola",
        "classroom": "sala de aula",
        "bathroom": "banheiro",
        "house": "casa",
        "building": "edifício",
        "sidewalk": "calçada",

        "sunny": "ensolarado",
        "cloudy": "nublado",
        "rainy": "chuvoso",
        "foggy": "com neblina",
        "snowy": "com neve",
        "clear_weather": "tempo limpo",
        "low_light": "pouca iluminação",
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
            self.LABEL_TO_PT.get(
                value,
                value.replace("_", " ")
            )
            for value in values
        ]

    def build(
        self,
        data: Dict[str, Any],
    ) -> str:

        subjects = list(dict.fromkeys(
            data.get("subjects", [])
        ))

        actions = list(dict.fromkeys(
            data.get("actions", [])
        ))

        objects = list(dict.fromkeys(
            data.get("objects", [])
        ))

        environment = list(dict.fromkeys(
            data.get("environment", [])
        ))

        attributes = list(dict.fromkeys(
            data.get("attributes", [])
        ))

        if "person" in subjects and any(
            label in subjects
            for label in [
                "woman", "man", "boy", "girl", "child"
            ]
        ):
            subjects.remove("person")

        if "moving" in actions and any(
            label in actions
            for label in [
                "walking", "running", "jumping",
                "dancing", "swimming"
            ]
        ):
            actions.remove("moving")

        if "still" in actions and "standing" in actions:
            actions.remove("still")

        if "outdoor" in environment and any(
            label in environment
            for label in [
                "street", "road", "park", "forest",
                "beach", "city"
            ]
        ):
            environment.remove("outdoor")

        if "sunny" in attributes and "rainy" in attributes:
            attributes.remove("sunny")

        if "clear_weather" in attributes and "rainy" in attributes:
            attributes.remove("clear_weather")

        facts = {
            "sujeitos": self.translate_values(
                subjects[:2]
            ),
            "acoes": self.translate_values(
                actions[:2]
            ),
            "objetos": self.translate_values(
                objects[:3]
            ),
            "ambiente": self.translate_values(
                environment[:2]
            ),
            "caracteristicas": self.translate_values(
                attributes[:3]
            ),
        }

        facts = {
            key: value
            for key, value in facts.items()
            if value
        }

        payload = json.dumps(
            facts,
            ensure_ascii=False,
        )
        
        
        history = data.get(
            "narrative_history", []
        )

        history_lines = []

        for record in history[-3:]:
            description = record.get(
                "description", ""
            )

            previous_labels = record.get(
                "labels", []
            )

            shared_labels = [
                label
                for label in previous_labels
                if label in data.get("labels", [])
            ]

            
            continuity_anchors = {
                "woman", "man", "boy", "girl",
                "child", "dog", "cat"
            }

            if not (
                set(shared_labels)
                & continuity_anchors
            ):
                continue

            history_lines.append(
                {
                    "descricao_anterior": description,
                    "fatos_em_comum": self.translate_values(
                        shared_labels[:5]
                    ),
                }
            )

        history_payload = json.dumps(
            history_lines,
            ensure_ascii=False,
        )

        
        return (
            "Você é o narrador de audiodescrição "
            "do See2Sound.\n"
            "Escreva UMA frase curta e natural "
            "em português brasileiro.\n"
            "Utilize somente os fatos da cena atual "
            "para afirmar o que está visível.\n"
            "O histórico é apenas uma referência "
            "de continuidade linguística, não uma "
            "fonte de novos fatos.\n"
            "Não copie fatos antigos que não estejam "
            "confirmados na cena atual.\n"
            "Evite repetir descrições anteriores.\n"
            "Não invente ações, relações, objetos, "
            "emoções ou mudanças de cenário.\n"
            "Não use primeira pessoa.\n"
            "Não associe um objeto a uma pessoa "
            "sem confirmação dessa relação.\n"
            "Não escreva JSON, títulos, explicações "
            "ou listas.\n"
            f"HISTÓRICO:\n{history_payload}\n"
            f"FATOS ATUAIS:\n{payload}\n"
            "AUDIODESCRIÇÃO:"
        )
