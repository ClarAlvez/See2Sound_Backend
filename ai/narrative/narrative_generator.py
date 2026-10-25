import re
import unicodedata

from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from ai.narrative.data_models import (
    NarrativeInput,
    NarrativeOutput,
    SpectraScene,
)

from ai.narrative.llm_client import (
    LlamaCppClient,
)


class LLMNarrativeGenerator:
    # ========================================================
    # Sujeitos
    # ========================================================

    SUBJECTS = {
        "woman": "Uma mulher",
        "man": "Um homem",
        "girl": "Uma menina",
        "boy": "Um menino",
        "child": "Uma criança",
        "person": "Uma pessoa",

        "dog": "Um cachorro",
        "cat": "Um gato",
        "bird": "Um pássaro",
        "horse": "Um cavalo",
        "sheep": "Uma ovelha",
        "cow": "Uma vaca",
        "fish": "Um peixe",
        "animal": "Um animal",
    }

    SUBJECT_PRIORITY = [
        "woman",
        "man",
        "girl",
        "boy",
        "child",
        "person",

        "dog",
        "cat",
        "bird",
        "horse",
        "sheep",
        "cow",
        "fish",
        "animal",
    ]

    HUMAN_SUBJECTS = {
        "person",
        "woman",
        "man",
        "boy",
        "girl",
        "child",
    }

    # ========================================================
    # Ações
    # ========================================================

    ACTIONS = {
        "standing": "está em pé",
        "sitting": "está sentada",
        "lying_down": "está deitada",
        "crouching": "está agachada",
        "arms_raised": (
            "mantém os braços erguidos"
        ),

        "walking": "caminha",
        "running": "corre",
        "jumping": "salta",
        "cycling": "anda de bicicleta",
        "driving": "dirige",
        "swimming": "nada",
        "climbing": "escala",

        "dancing": "dança",
        "playing": "brinca",
        "working": "trabalha",
        "eating": "come",
        "drinking": "bebe",
        "exercising": "se exercita",
        "sports": (
            "pratica uma atividade esportiva"
        ),
        "instrument_playing": (
            "toca um instrumento"
        ),
        "phone_use": (
            "usa um telefone"
        ),
        "computer_use": (
            "usa um computador"
        ),
        "reading": "lê",
        "writing": "escreve",
        "cooking": "cozinha",
        "cleaning": "realiza uma limpeza",
        "makeup": (
            "faz uso de maquiagem"
        ),
        "grooming": (
            "cuida da própria aparência"
        ),
        "talking": "fala",
        "reaching": (
            "estende o braço"
        ),
        "throwing": "arremessa algo",
        "carrying": "carrega algo",
        "radio_use": (
            "usa um rádio"
        ),
        "ball_sport": (
            "pratica um esporte com bola"
        ),
        "racket_sport": (
            "pratica um esporte com raquete"
        ),
        "martial_activity": (
            "realiza uma atividade marcial"
        ),
        "water_activity": (
            "realiza uma atividade na água"
        ),

        "still": (
            "permanece parada"
        ),
        "moving": (
            "está em movimento"
        ),
        "fast_motion": (
            "se move rapidamente"
        ),
        "falling": "cai",
    }

    # ========================================================
    # Aparência
    # ========================================================

    APPEARANCE = {
        "short_hair": (
            "com cabelos curtos"
        ),
        "long_hair": (
            "com cabelos longos"
        ),
        "bald_hair": "sem cabelo",

        "black_hair": (
            "com cabelos pretos"
        ),
        "blonde_hair": (
            "com cabelos loiros"
        ),
        "brown_hair": (
            "com cabelos castanhos"
        ),
        "gray_hair": (
            "com cabelos grisalhos"
        ),

        "straight_hair": (
            "com cabelos lisos"
        ),
        "wavy_hair": (
            "com cabelos ondulados"
        ),
        "bangs_hair": (
            "com franja"
        ),
        "receding_hairline": (
            "com entradas no cabelo"
        ),

        "black_clothes": (
            "com roupas pretas"
        ),
        "white_clothes": (
            "com roupas brancas"
        ),
        "red_clothes": (
            "com roupas vermelhas"
        ),
        "blue_clothes": (
            "com roupas azuis"
        ),
        "green_clothes": (
            "com roupas verdes"
        ),
        "yellow_clothes": (
            "com roupas amarelas"
        ),
        "brown_clothes": (
            "com roupas marrons"
        ),
        "gray_clothes": (
            "com roupas cinzas"
        ),
        "orange_clothes": (
            "com roupas laranjas"
        ),
        "pink_clothes": (
            "com roupas rosas"
        ),
        "purple_clothes": (
            "com roupas roxas"
        ),

        "dress": (
            "usando um vestido"
        ),

        "glasses": (
            "usando óculos"
        ),
        "hat": (
            "usando um chapéu"
        ),
        "backpack": (
            "com uma mochila"
        ),
        "bag": (
            "com uma bolsa"
        ),
    }

    # ========================================================
    # Ambientes
    # ========================================================

    ENVIRONMENTS = {
        "street": "em uma rua",
        "road": "em uma estrada",
        "city": "em uma área urbana",
        "desert": "em um deserto",
        "beach": "em uma praia",
        "ocean": (
            "em uma cena com oceano"
        ),
        "forest": "em uma floresta",
        "mountain": (
            "em uma região montanhosa"
        ),
        "park": "em um parque",
        "field": "em um campo",

        "school": "em uma escola",
        "classroom": (
            "em uma sala de aula"
        ),
        "kitchen": "em uma cozinha",
        "bedroom": "em um quarto",
        "living_room": (
            "em uma sala de estar"
        ),

        "office_room": (
            "em um escritório"
        ),
        "office_cubicles": (
            "em um escritório com cabines"
        ),
        "home_office": (
            "em um escritório doméstico"
        ),
        "conference_room": (
            "em uma sala de reunião"
        ),
        "office": (
            "em um escritório"
        ),

        "restaurant_indoor": (
            "dentro de um restaurante"
        ),
        "fastfood_restaurant": (
            "em um restaurante de "
            "comida rápida"
        ),
        "cafeteria": (
            "em uma cafeteria"
        ),
        "dining_room": (
            "em uma sala de jantar"
        ),
        "restaurant_patio": (
            "na área externa "
            "de um restaurante"
        ),
        "restaurant": (
            "em um restaurante"
        ),

        "store": "em uma loja",
        "hospital": "em um hospital",
        "sports_field": (
            "em um campo esportivo"
        ),

        "room": (
            "em um ambiente interno"
        ),
        "indoor": (
            "em um ambiente interno"
        ),
        "outdoor": "ao ar livre",
    }

    ENVIRONMENT_BASE_DESCRIPTIONS = {
        "street": (
            "Uma rua aparece na cena"
        ),
        "road": (
            "Uma estrada aparece na cena"
        ),
        "city": (
            "Uma área urbana "
            "aparece na cena"
        ),
        "desert": (
            "A cena mostra um deserto"
        ),
        "beach": (
            "A cena mostra uma praia"
        ),
        "ocean": (
            "O oceano aparece no cenário"
        ),
        "forest": (
            "A cena mostra uma floresta"
        ),
        "mountain": (
            "A cena mostra uma "
            "região montanhosa"
        ),
        "park": (
            "A cena mostra um parque"
        ),
        "field": (
            "A cena mostra um campo"
        ),

        "school": (
            "A cena mostra uma escola"
        ),
        "classroom": (
            "A cena mostra uma "
            "sala de aula"
        ),
        "kitchen": (
            "A cena mostra uma cozinha"
        ),
        "bedroom": (
            "A cena mostra um quarto"
        ),
        "living_room": (
            "A cena mostra uma "
            "sala de estar"
        ),

        "office_room": (
            "A cena mostra um escritório"
        ),
        "office_cubicles": (
            "A cena mostra um escritório "
            "com cabines"
        ),
        "home_office": (
            "A cena mostra um "
            "escritório doméstico"
        ),
        "conference_room": (
            "A cena mostra uma "
            "sala de reunião"
        ),
        "office": (
            "A cena mostra um escritório"
        ),

        "restaurant_indoor": (
            "A cena mostra o interior "
            "de um restaurante"
        ),
        "fastfood_restaurant": (
            "A cena mostra um restaurante "
            "de comida rápida"
        ),
        "cafeteria": (
            "A cena mostra uma cafeteria"
        ),
        "dining_room": (
            "A cena mostra uma "
            "sala de jantar"
        ),
        "restaurant_patio": (
            "A cena mostra a área externa "
            "de um restaurante"
        ),
        "restaurant": (
            "A cena mostra um restaurante"
        ),

        "store": (
            "A cena mostra uma loja"
        ),
        "hospital": (
            "A cena mostra um hospital"
        ),
        "sports_field": (
            "A cena mostra um "
            "campo esportivo"
        ),

        "room": (
            "A cena mostra um "
            "ambiente interno"
        ),
        "indoor": (
            "A cena mostra um "
            "ambiente interno"
        ),
        "outdoor": (
            "A cena mostra um "
            "ambiente externo"
        ),
    }

    # Mais específicos vêm antes de
    # indoor/outdoor.
    ENVIRONMENT_PRIORITY = [
        "street",
        "road",
        "city",
        "desert",
        "beach",
        "ocean",
        "forest",
        "mountain",
        "park",
        "field",

        "school",
        "classroom",
        "kitchen",
        "bedroom",
        "living_room",

        "office_room",
        "office_cubicles",
        "home_office",
        "conference_room",
        "office",

        "restaurant_indoor",
        "fastfood_restaurant",
        "cafeteria",
        "dining_room",
        "restaurant_patio",
        "restaurant",

        "store",
        "hospital",
        "sports_field",

        "room",
        "indoor",
        "outdoor",
    ]

    # ========================================================
    # Objetos
    # ========================================================

    OBJECTS = {
        "chair": "uma cadeira",
        "table": "uma mesa",
        "sofa": "um sofá",
        "bed": "uma cama",
        "toilet": "um vaso sanitário",
        "door": "uma porta",
        "window": "uma janela",
        "mirror": "um espelho",
        "desk": "uma escrivaninha",
        "shelf": "uma prateleira",
        "cabinet": "um armário",
        "lamp": "uma luminária",

        "phone": "um telefone",
        "computer": "um computador",
        "screen": "uma tela",
        "television": "uma televisão",
        "keyboard": "um teclado",
        "mouse": "um mouse",
        "remote": "um controle remoto",
        "camera": "uma câmera",
        "headphones": "um fone de ouvido",
        "microphone": "um microfone",

        "car": "um carro",
        "bicycle": "uma bicicleta",
        "motorcycle": "uma motocicleta",
        "bus": "um ônibus",
        "truck": "um caminhão",
        "train": "um trem",
        "boat": "um barco",
        "airplane": "um avião",

        "food": "comida",
        "fruit": "uma fruta",
        "cup": "um copo",
        "bottle": "uma garrafa",
        "bowl": "uma tigela",
        "plate": "um prato",
        "fork": "um garfo",
        "spoon": "uma colher",
        "knife": "uma faca",

        "bag": "uma bolsa",
        "backpack": "uma mochila",
        "handbag": "uma bolsa de mão",
        "suitcase": "uma mala",
        "umbrella": "um guarda-chuva",
        "glasses": "óculos",
        "watch": "um relógio",
        "hat": "um chapéu",

        "book": "um livro",
        "paper": "um papel",
        "letter": "uma carta",
        "document": "um documento",
        "photo": "uma fotografia",
        "newspaper": "um jornal",
        "poster": "um cartaz",

        "ball": "uma bola",
        "toy": "um brinquedo",
        "kite": "uma pipa",
        "skateboard": "um skate",
        "surfboard": "uma prancha",
        "sports_racket": "uma raquete",

        "box": "uma caixa",
        "key": "uma chave",
        "weapon": "uma arma",
        "blood": "sangue",
        "musical_instrument": (
            "um instrumento musical"
        ),
        "dice": "um dado",
        "miniature": "uma miniatura",
        "board_game": (
            "um jogo de tabuleiro"
        ),
        "subtitles": "legendas",
        "on_screen_text": (
            "texto na tela"
        ),
    }

    # ========================================================
    # Atmosfera / horário / iluminação
    # ========================================================

    WEATHER = {
        "clear_weather": (
            "com tempo aberto"
        ),
        "sunny": (
            "sob tempo ensolarado"
        ),
        "cloudy": (
            "sob tempo nublado"
        ),
        "foggy": (
            "em meio à neblina"
        ),
        "rainy": (
            "em meio à chuva"
        ),
        "snowy": (
            "em meio à neve"
        ),
    }

    WEATHER_PRIORITY = [
        "rainy",
        "snowy",
        "foggy",
        "cloudy",
        "sunny",
        "clear_weather",
    ]

    TIME = {
        "day": "durante o dia",
        "night": "à noite",
        "dawn_dusk": (
            "durante a transição "
            "entre dia e noite"
        ),
    }

    TIME_PRIORITY = [
        "night",
        "dawn_dusk",
        "day",
    ]

    LIGHTING = {
        "bright": (
            "com iluminação intensa"
        ),
        "low_light": (
            "com pouca iluminação"
        ),
    }

    LIGHTING_PRIORITY = [
        "low_light",
        "bright",
    ]

    # ========================================================
    # Inicialização
    # ========================================================

    def __init__(
        self,
        model_path: str = (
            "data/models/qwen/"
            "qwen2.5-3b-instruct-"
            "q4_k_m.gguf"
        ),
        similarity_threshold: float = 0.75,
        n_ctx: int = 4096,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0,
        use_llm: bool = True,
    ):
        self.use_llm = (
            use_llm
        )

        self.similarity_threshold = (
            similarity_threshold
        )

        self.client = None

        if self.use_llm:
            print(
                "[Narrative] Carregando "
                "modelo de linguagem:"
            )

            print(
                "[Narrative]",
                model_path,
            )

            self.client = (
                LlamaCppClient(
                    model_path=(
                        model_path
                    ),
                    n_ctx=n_ctx,
                    n_threads=n_threads,
                    n_gpu_layers=(
                        n_gpu_layers
                    ),
                    temperature=0.1,
                    top_p=0.8,
                    max_tokens=80,
                    verbose=False,
                )
            )

    # ========================================================
    # Utilidades
    # ========================================================

    def _normalize(
        self,
        text: str,
    ) -> str:
        text = (
            unicodedata.normalize(
                "NFD",
                text.lower(),
            )
        )

        text = "".join(
            character
            for character in text
            if unicodedata.category(
                character
            ) != "Mn"
        )

        return re.sub(
            r"\s+",
            " ",
            re.sub(
                r"[^\w\s]",
                " ",
                text,
            ),
        ).strip()

    def _get_confidence(
        self,
        label: str,
        confidence: Dict[str, Any],
    ) -> Optional[float]:
        if not isinstance(
            confidence,
            dict,
        ):
            return None

        scores = []

        for (
            key,
            value,
        ) in confidence.items():
            if (
                key == label
                or key.endswith(
                    "." + label
                )
            ):
                try:
                    scores.append(
                        float(value)
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    pass

        if not scores:
            return None

        return max(
            scores
        )

    def _accepted(
        self,
        label: str,
        labels: set,
        confidence: Dict[str, Any],
        threshold: float,
    ) -> bool:
        if label not in labels:
            return False

        score = self._get_confidence(
            label,
            confidence,
        )

        # Se a label já veio filtrada pela
        # Spectra e não possui score individual,
        # ela é considerada válida.
        if score is None:
            return True

        return (
            score
            >= threshold
        )

    # ========================================================
    # Seleção dos fatos
    # ========================================================

    def _select_facts(
        self,
        narrative_input: NarrativeInput,
    ) -> Dict[str, Any]:
        labels = {
            label.strip().lower()
            for label
            in narrative_input.labels
            if isinstance(
                label,
                str,
            )
            and label.strip()
        }

        confidence = (
            narrative_input.confidence
            or {}
        )

        # -----------------------------------------------
        # Sujeito
        #
        # Alinhado ao threshold do modelo Person.
        # -----------------------------------------------

        subject = None

        for label in (
            self.SUBJECT_PRIORITY
        ):
            if self._accepted(
                label=label,
                labels=labels,
                confidence=confidence,
                threshold=0.50,
            ):
                subject = label
                break

        # -----------------------------------------------
        # Ação
        #
        # Só relacionamos ação quando existe
        # sujeito detectado.
        # -----------------------------------------------

        action = None

        candidates = []

        if subject is not None:
            candidates = []

            for label in (
                self.ACTIONS
            ):
                if self._accepted(
                    label=label,
                    labels=labels,
                    confidence=confidence,
                    threshold=0.60,
                ):
                    candidates.append(
                        label
                    )

            
        if candidates:
            context = (
                narrative_input.context
                or {}
            )

            preferred_action = context.get(
                "preferred_action"
            )

            # Só aceita a preferência quando
            # a ação existe entre as candidatas
            # já validadas pela Narrative.
            if preferred_action in candidates:
                action = preferred_action

                print(
                    "[Narrative] Ação priorizada "
                    "pela fusão:",
                    action,
                )

            else:
                # Comportamento original da Spectra.
                action = max(
                    candidates,
                    key=lambda label: (
                        self._get_confidence(
                            label,
                            confidence,
                        )
                        or 0.0
                    ),
                )


        # -----------------------------------------------
        # Aparência
        # -----------------------------------------------

        appearance = []

        if subject in (
            self.HUMAN_SUBJECTS
        ):
            candidates = []

            for label in (
                self.APPEARANCE
            ):
                if self._accepted(
                    label=label,
                    labels=labels,
                    confidence=confidence,
                    threshold=0.50,
                ):
                    candidates.append(
                        label
                    )

            candidates.sort(
                key=lambda label: (
                    self._get_confidence(
                        label,
                        confidence,
                    )
                    or 0.0
                ),
                reverse=True,
            )

            appearance = (
                candidates[:2]
            )

            # Contradições básicas.
            if (
                "short_hair"
                in appearance
                and "long_hair"
                in appearance
            ):
                best_hair = max(
                    [
                        "short_hair",
                        "long_hair",
                    ],
                    key=lambda label: (
                        self._get_confidence(
                            label,
                            confidence,
                        )
                        or 0.0
                    ),
                )

                appearance = [
                    value
                    for value
                    in appearance
                    if value not in {
                        "short_hair",
                        "long_hair",
                    }
                    or value
                    == best_hair
                ]

        # -----------------------------------------------
        # Ambiente
        #
        # Labels específicas têm preferência sobre
        # indoor/outdoor.
        # -----------------------------------------------

        environment = None

        for label in (
            self.ENVIRONMENT_PRIORITY
        ):
            if self._accepted(
                label=label,
                labels=labels,
                confidence=confidence,
                threshold=0.45,
            ):
                environment = label
                break

        # -----------------------------------------------
        # Objetos
        #
        # Alinhado ao threshold Spectra Object = 0.35.
        # -----------------------------------------------

        object_candidates = []

        for label in (
            self.OBJECTS
        ):
            if self._accepted(
                label=label,
                labels=labels,
                confidence=confidence,
                threshold=0.35,
            ):
                object_candidates.append(
                    label
                )

        object_candidates.sort(
            key=lambda label: (
                self._get_confidence(
                    label,
                    confidence,
                )
                or 0.0
            ),
            reverse=True,
        )

        objects = (
            object_candidates[:2]
        )

        # -----------------------------------------------
        # Clima
        # -----------------------------------------------

        atmosphere = None

        for label in (
            self.WEATHER_PRIORITY
        ):
            if self._accepted(
                label=label,
                labels=labels,
                confidence=confidence,
                threshold=0.50,
            ):
                atmosphere = label
                break

        # -----------------------------------------------
        # Horário
        # -----------------------------------------------

        time_of_day = None

        for label in (
            self.TIME_PRIORITY
        ):
            if self._accepted(
                label=label,
                labels=labels,
                confidence=confidence,
                threshold=0.50,
            ):
                time_of_day = label
                break

        # -----------------------------------------------
        # Iluminação
        # -----------------------------------------------

        lighting = None

        for label in (
            self.LIGHTING_PRIORITY
        ):
            if self._accepted(
                label=label,
                labels=labels,
                confidence=confidence,
                threshold=0.50,
            ):
                lighting = label
                break

        return {
            "subject": subject,
            "action": action,
            "appearance": (
                appearance
            ),
            "environment": (
                environment
            ),
            "objects": objects,
            "atmosphere": (
                atmosphere
            ),
            "time": (
                time_of_day
            ),
            "lighting": (
                lighting
            ),
        }

    # ========================================================
    # Formatação
    # ========================================================

    def _subject_reference(
        self,
        subject: str,
    ) -> str:
        references = {
            "woman": "A mulher",
            "man": "O homem",
            "girl": "A menina",
            "boy": "O menino",
            "child": "A criança",
            "person": "A pessoa",

            "dog": "O cachorro",
            "cat": "O gato",
            "bird": "O pássaro",
            "horse": "O cavalo",
            "sheep": "A ovelha",
            "cow": "A vaca",
            "fish": "O peixe",
            "animal": "O animal",
        }

        return references.get(
            subject,
            self.SUBJECTS.get(
                subject,
                "A pessoa",
            ),
        )

    def _action_text(
        self,
        subject: str,
        action: str,
    ) -> str:
        text = self.ACTIONS[
            action
        ]

        masculine_subjects = {
            "man",
            "boy",
            "dog",
            "cat",
            "bird",
            "horse",
            "fish",
            "animal",
        }

        if action == "sitting":
            if (
                subject
                in masculine_subjects
            ):
                return (
                    "está sentado"
                )

            return "está sentada"

        if action == "lying_down":
            if (
                subject
                in masculine_subjects
            ):
                return (
                    "está deitado"
                )

            return "está deitada"

        if action == "crouching":
            if (
                subject
                in masculine_subjects
            ):
                return (
                    "está agachado"
                )

            return (
                "está agachada"
            )

        if action == "still":
            if (
                subject
                in masculine_subjects
            ):
                return (
                    "permanece parado"
                )

            return (
                "permanece parada"
            )

        return text

    def _append_scene_modifiers(
        self,
        text: str,
        facts: Dict[str, Any],
        include_atmosphere: bool = True,
        include_time: bool = True,
        include_lighting: bool = True,
    ) -> str:
        modifiers = []

        atmosphere = facts.get(
            "atmosphere"
        )

        time_of_day = facts.get(
            "time"
        )

        lighting = facts.get(
            "lighting"
        )

        if (
            include_atmosphere
            and atmosphere
            in self.WEATHER
        ):
            modifiers.append(
                self.WEATHER[
                    atmosphere
                ]
            )

        if (
            include_time
            and time_of_day
            in self.TIME
        ):
            modifiers.append(
                self.TIME[
                    time_of_day
                ]
            )

        if (
            include_lighting
            and lighting
            in self.LIGHTING
        ):
            modifiers.append(
                self.LIGHTING[
                    lighting
                ]
            )

        if modifiers:
            text += (
                ", "
                + ", ".join(
                    modifiers
                )
            )

        return text

    def _build_base_description(
        self,
        facts: Dict[str, Any],
    ) -> str:
        subject = facts.get(
            "subject"
        )

        action = facts.get(
            "action"
        )

        appearance = facts.get(
            "appearance",
            [],
        )

        environment = facts.get(
            "environment"
        )

        objects = facts.get(
            "objects",
            [],
        )

        atmosphere = facts.get(
            "atmosphere"
        )

        time_of_day = facts.get(
            "time"
        )

        lighting = facts.get(
            "lighting"
        )

        # -----------------------------------------------
        # Cena com sujeito
        # -----------------------------------------------

        if subject:
            text = (
                self.SUBJECTS[
                    subject
                ]
            )

            if appearance:
                text += (
                    " "
                    + " e ".join(
                        self.APPEARANCE[
                            label
                        ]
                        for label
                        in appearance
                        if label
                        in self.APPEARANCE
                    )
                )

            if action:
                text += (
                    " "
                    + self._action_text(
                        subject,
                        action,
                    )
                )

            else:
                text += " aparece"

            if (
                environment
                and environment
                in self.ENVIRONMENTS
            ):
                text += (
                    " "
                    + self.ENVIRONMENTS[
                        environment
                    ]
                )

            if objects:
                object_names = [
                    self.OBJECTS[
                        item
                    ]
                    for item
                    in objects
                    if item
                    in self.OBJECTS
                ]

                if object_names:
                    if len(
                        object_names
                    ) == 1:
                        text += (
                            ", com "
                            + object_names[0]
                            + " também visível"
                        )

                    else:
                        text += (
                            ", com "
                            + " e ".join(
                                object_names
                            )
                            + " também visíveis"
                        )

            text = (
                self._append_scene_modifiers(
                    text,
                    facts,
                )
            )

            return (
                text.rstrip(
                    "."
                )
                + "."
            )

        # -----------------------------------------------
        # Cena sem sujeito
        # -----------------------------------------------

        text = ""

        if (
            environment
            and environment
            in self.ENVIRONMENT_BASE_DESCRIPTIONS
        ):
            text = (
                self.ENVIRONMENT_BASE_DESCRIPTIONS[
                    environment
                ]
            )

        elif objects:
            first_object = (
                self.OBJECTS.get(
                    objects[0],
                    objects[0],
                )
            )

            text = (
                first_object[0]
                .upper()
                + first_object[1:]
                + " aparece na cena"
            )

        elif (
            atmosphere
            or time_of_day
            or lighting
        ):
            text = (
                "A cena apresenta "
                "mudanças visuais"
            )

        if not text:
            return ""

        # Objetos podem complementar
        # uma descrição ambiental.
        if (
            environment
            and objects
        ):
            object_names = [
                self.OBJECTS[
                    item
                ]
                for item
                in objects
                if item
                in self.OBJECTS
            ]

            if object_names:
                text += (
                    ", com "
                    + " e ".join(
                        object_names
                    )
                    + " visíveis"
                )

        text = (
            self._append_scene_modifiers(
                text,
                facts,
            )
        )

        return (
            text.rstrip(
                "."
            )
            + "."
        )

    # ========================================================
    # Reescrita segura pelo Qwen
    # ========================================================

    def _safe_llm_rewrite(
        self,
        base_description: str,
    ) -> str:
        if (
            not self.use_llm
            or self.client is None
        ):
            return (
                base_description
            )

        prompt = (
            "Reescreva a frase abaixo em "
            "português brasileiro natural, "
            "adequado para audiodescrição. "
            "Não acrescente sujeitos, ações, "
            "objetos, ambientes, emoções ou "
            "qualquer informação nova. "
            "Preserve todos os fatos existentes. "
            "Use apenas uma frase curta. "
            "Retorne somente a frase.\n\n"
            f"Frase: {base_description}"
        )

        try:
            raw = (
                self.client.generate(
                    prompt
                )
            )

            candidate = (
                raw.strip()
            )

            if not candidate:
                return (
                    base_description
                )

            if any(
                character in candidate
                for character
                in "{}[]\n"
            ):
                return (
                    base_description
                )

            # Evita respostas com múltiplas
            # frases ou explicações.
            if len(
                re.findall(
                    r"[.!?]",
                    candidate,
                )
            ) > 1:
                return (
                    base_description
                )

            base_words = (
                self._normalize(
                    base_description
                )
                .split()
            )

            candidate_words = (
                self._normalize(
                    candidate
                )
                .split()
            )

            allowed_connectors = {
                "a",
                "o",
                "as",
                "os",
                "um",
                "uma",
                "uns",
                "umas",
                "de",
                "do",
                "da",
                "dos",
                "das",
                "em",
                "no",
                "na",
                "nos",
                "nas",
                "e",
                "com",
                "ao",
                "aos",
                "sob",
                "durante",
                "tambem",
                "cena",
            }

            allowed_words = (
                set(
                    base_words
                )
                | allowed_connectors
            )

            # Se o Qwen introduzir vocabulário
            # factual novo, usamos a frase
            # determinística original.
            if any(
                word
                not in allowed_words
                for word
                in candidate_words
            ):
                return (
                    base_description
                )

            return candidate

        except Exception as error:
            print(
                "[Narrative LLM] "
                "Reformulação ignorada:",
                error,
            )

            return (
                base_description
            )

    # ========================================================
    # Uma cena
    # ========================================================

    def generate(
        self,
        narrative_input: NarrativeInput,
    ) -> NarrativeOutput:
        labels = list(
            dict.fromkeys(
                label.strip().lower()
                for label
                in narrative_input.labels
                if isinstance(
                    label,
                    str,
                )
                and label.strip()
            )
        )

        facts = (
            self._select_facts(
                narrative_input
            )
        )

        base_description = (
            self._build_base_description(
                facts
            )
        )

        if not base_description:
            return NarrativeOutput(
                description="",
                start_time=(
                    narrative_input.start_time
                ),
                end_time=(
                    narrative_input.end_time
                ),
                labels=labels,
                scene_context=facts,
                skipped=True,
                skip_reason=(
                    "Fatos insuficientes "
                    "para uma descrição "
                    "confiável."
                ),
            )

        return NarrativeOutput(
            description=(
                base_description
            ),
            start_time=(
                narrative_input.start_time
            ),
            end_time=(
                narrative_input.end_time
            ),
            labels=labels,
            scene_context=facts,
            skipped=False,
            skip_reason=None,
            fidelity_warnings=[],
        )

    # ========================================================
    # Mudanças entre cenas
    # ========================================================

    def _build_delta_description(
        self,
        current: Dict[str, Any],
        previous: Optional[
            Dict[str, Any]
        ],
    ) -> str:
        # Primeira descrição:
        # descreve tudo que for seguro.
        if previous is None:
            return (
                self._build_base_description(
                    current
                )
            )

        subject = current.get(
            "subject"
        )

        old_subject = previous.get(
            "subject"
        )

        environment = current.get(
            "environment"
        )

        old_environment = previous.get(
            "environment"
        )

        objects = current.get(
            "objects",
            [],
        )

        old_objects = previous.get(
            "objects",
            [],
        )

        atmosphere = current.get(
            "atmosphere"
        )

        old_atmosphere = previous.get(
            "atmosphere"
        )

        time_of_day = current.get(
            "time"
        )

        old_time = previous.get(
            "time"
        )

        lighting = current.get(
            "lighting"
        )

        old_lighting = previous.get(
            "lighting"
        )

        action = current.get(
            "action"
        )

        old_action = previous.get(
            "action"
        )

        appearance = current.get(
            "appearance",
            [],
        )

        old_appearance = previous.get(
            "appearance",
            [],
        )

        # -----------------------------------------------
        # Cena sem sujeito
        # -----------------------------------------------

        if subject is None:
            subject_disappeared = (
                old_subject is not None
            )

            environment_changed = (
                environment
                != old_environment
            )

            new_objects = [
                item
                for item
                in objects
                if item
                not in old_objects
            ]

            atmosphere_changed = (
                atmosphere
                != old_atmosphere
            )

            time_changed = (
                time_of_day
                != old_time
            )

            lighting_changed = (
                lighting
                != old_lighting
            )

            if not any(
                [
                    subject_disappeared,
                    environment_changed,
                    new_objects,
                    atmosphere_changed,
                    time_changed,
                    lighting_changed,
                ]
            ):
                return ""

            return (
                self._build_base_description(
                    current
                )
            )

        # -----------------------------------------------
        # Um sujeito apareceu onde antes
        # não havia nenhum.
        # -----------------------------------------------

        if old_subject is None:
            return (
                self._build_base_description(
                    current
                )
            )

        same_category = (
            subject
            == old_subject
            or (
                subject
                in self.HUMAN_SUBJECTS
                and old_subject
                in self.HUMAN_SUBJECTS
            )
        )

        if not same_category:
            return (
                self._build_base_description(
                    current
                )
            )

        # -----------------------------------------------
        # Mudanças relevantes
        # -----------------------------------------------

        action_changed = (
            action
            != old_action
        )

        environment_changed = (
            environment
            != old_environment
        )

        atmosphere_changed = (
            atmosphere
            != old_atmosphere
        )

        time_changed = (
            time_of_day
            != old_time
        )

        lighting_changed = (
            lighting
            != old_lighting
        )

        new_appearance = [
            item
            for item
            in appearance
            if item
            not in old_appearance
        ]

        new_objects = [
            item
            for item
            in objects
            if item
            not in old_objects
        ]

        subject_refined = (
            old_subject == "person"
            and subject
            in {
                "woman",
                "man",
                "girl",
                "boy",
                "child",
            }
        )

        if not any(
            [
                action_changed,
                environment_changed,
                atmosphere_changed,
                time_changed,
                lighting_changed,
                new_appearance,
                new_objects,
                subject_refined,
            ]
        ):
            return ""

        # Para evitar inconsistências de
        # continuidade, reconstruímos a frase
        # somente com os fatos confirmados da
        # cena atual.
        return (
            self._build_base_description(
                current
            )
        )

    # ========================================================
    # Lote / memória
    # ========================================================

    def generate_batch(
        self,
        inputs: List[
            NarrativeInput
        ],
        skip_similar_labels: bool = True,
        skip_similar_text: bool = True,
    ) -> List[
        NarrativeOutput
    ]:
        outputs = []

        last_narrated_facts = None
        last_narrated_end = None
        last_description = ""

        max_memory_gap = 8.0
        time_tolerance = 0.15

        for (
            index,
            item,
        ) in enumerate(
            inputs
        ):
            output = (
                self.generate(
                    item
                )
            )

            current_facts = (
                output.scene_context
            )

            if output.skipped:
                outputs.append(
                    output
                )

                continue

            gap = None

            if (
                last_narrated_end
                is not None
                and item.start_time
                is not None
            ):
                gap = (
                    float(
                        item.start_time
                    )
                    - float(
                        last_narrated_end
                    )
                )

            memory_valid = (
                last_narrated_facts
                is not None
                and gap
                is not None
                and (
                    -time_tolerance
                    <= gap
                    <= max_memory_gap
                )
            )

            previous = (
                last_narrated_facts
                if memory_valid
                else None
            )

            print(
                "\n[Narrative MEMORY] "
                f"Cena {index + 1}"
            )

            print(
                "  Início:",
                item.start_time,
            )

            print(
                "  Último fim:",
                last_narrated_end,
            )

            print(
                "  Diferença:",
                gap,
            )

            print(
                "  Memória válida:",
                memory_valid,
            )

            description = (
                self._build_delta_description(
                    current=(
                        current_facts
                    ),
                    previous=previous,
                )
            )

            if not description:
                output.skipped = True

                output.description = ""

                output.skip_reason = (
                    "Nenhuma informação "
                    "visual nova relevante."
                )

                outputs.append(
                    output
                )

                continue

            if (
                skip_similar_text
                and last_description
                and self._normalize(
                    description
                )
                == self._normalize(
                    last_description
                )
            ):
                output.skipped = True

                output.description = ""

                output.skip_reason = (
                    "Descrição repetida."
                )

                outputs.append(
                    output
                )

                continue

            # -----------------------------------------------
            # Qwen
            # -----------------------------------------------

            if self.use_llm:
                description = (
                    self._safe_llm_rewrite(
                        description
                    )
                )

            output.description = (
                description
            )

            output.scene_context = {
                **current_facts,

                "narrative_mode": (
                    "delta"
                    if previous
                    else "initial"
                ),

                "reference_scene": (
                    last_narrated_end
                    if previous
                    else None
                ),

                "llm_enabled": (
                    self.use_llm
                ),
            }

            last_narrated_facts = dict(
                current_facts
            )

            last_narrated_end = (
                item.end_time
            )

            last_description = (
                description
            )

            outputs.append(
                output
            )

            print(
                "  Modo:",
                output.scene_context[
                    "narrative_mode"
                ],
            )

            print(
                "  Qwen:",
                self.use_llm,
            )


        return outputs

    # ========================================================
    # SpectraScene
    # ========================================================

    def generate_from_spectra_scenes(
        self,
        spectra_scenes: List[
            SpectraScene
        ],
    ) -> List[
        NarrativeOutput
    ]:
        return (
            self.generate_batch(
                [
                    NarrativeInput(
                        labels=(
                            scene.labels
                        ),
                        start_time=(
                            scene.start_time
                        ),
                        end_time=(
                            scene.end_time
                        ),
                        confidence=(
                            scene.confidence
                        ),
                        context=(
                            scene.context
                        ),
                    )
                    for scene
                    in spectra_scenes
                ]
            )
        )

    # ========================================================
    # Pipeline
    # ========================================================

    def generate_timeline_from_dicts(
        self,
        spectra_outputs: List[
            Dict[str, Any]
        ],
    ) -> List[
        Dict[str, Any]
    ]:
        inputs = [
            NarrativeInput(
                labels=(
                    item.get(
                        "labels",
                        [],
                    )
                ),
                start_time=(
                    item.get(
                        "start_time"
                    )
                ),
                end_time=(
                    item.get(
                        "end_time"
                    )
                ),
                confidence=(
                    item.get(
                        "confidence",
                        {},
                    )
                ),
                context=(
                    item.get(
                        "context",
                        {},
                    )
                ),
            )
            for item
            in spectra_outputs
        ]

        print(
            "[Narrative DEBUG] "
            f"Entradas recebidas: "
            f"{len(inputs)}"
        )

        outputs = (
            self.generate_batch(
                inputs
            )
        )

        valid_outputs = []

        for (
            index,
            output,
        ) in enumerate(
            outputs
        ):
            print(
                "\n[Narrative DEBUG] "
                f"Cena {index + 1}"
            )



            print(
                "  skipped:",
                output.skipped,
            )

            print(
                "  motivo:",
                output.skip_reason,
            )

            if not output.skipped:
                valid_outputs.append(
                    output.to_dict()
                )

        print(
            "[Narrative DEBUG] "
            "Descrições válidas: "
            f"{len(valid_outputs)}"
        )

        return valid_outputs