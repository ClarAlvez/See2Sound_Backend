import json
from typing import Any, Dict


class NarrativePromptBuilder:
    """
    Monta o prompt enviado ao modelo local.

    O LLM deve atuar como verbalizador controlado dos dados da Spectra,
    e não como um gerador livre de cenas.
    """

    def build(self, data: Dict[str, Any]) -> str:
        input_json = json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        )

        return """
Transforme os dados visuais abaixo em UMA única frase de audiodescrição
natural, curta e fiel.

REGRAS OBRIGATÓRIAS:

1. Use SOMENTE informações explicitamente presentes nos dados.
2. NÃO invente objetos, clima, iluminação, emoção, intenção ou detalhes de cenário.
3. Preserve gênero aparente quando disponível:
   - man = homem
   - woman = mulher
   - boy = menino
   - girl = menina
   - child = criança
4. Preserve exatamente a ação detectada:
   - running = correndo
   - walking = caminhando
   - sitting = sentado/sentada
   - standing = em pé
   - jumping = pulando
   - dancing = dançando
5. Nunca transforme "running" em "walking".
6. Nunca troque "man" por mulher ou pronomes femininos.
7. Nunca troque "woman" por homem ou pronomes masculinos.
8. Incorpore características visuais relevantes:
   - black_clothes = roupas pretas
   - white_clothes = roupas brancas
   - red_clothes = roupas vermelhas
   - glasses = óculos
   - short_hair = cabelo curto
   - long_hair = cabelo longo
9. Interprete ambientes:
   - field = campo
   - outdoor = ao ar livre
   - indoor = ambiente interno
10. Interprete movimento:
   - fast_motion = movimento rápido / rapidamente
   - slow_motion = movimento lento / lentamente
11. Não omita gênero, ação, roupa, acessórios ou ambiente quando esses dados existirem.
12. Não adicione cores que não tenham sido detectadas.
13. Não diga que existe sol, céu, árvores, prédios ou outros elementos se eles não aparecem nos dados.
14. Não use palavras técnicas como "label", "modelo", "detecção", "IA" ou "confidence".
15. Não diga "a imagem mostra" ou "a cena mostra".
16. Gere somente uma frase.
17. Escreva em português do Brasil.
18. Use no máximo 25 palavras.
19. Não use aspas.
20. Não explique sua resposta.

EXEMPLO:

Dados:
man, running, black_clothes, glasses, short_hair, field, outdoor, fast_motion

Resposta:
Um homem de cabelo curto, usando óculos e roupas pretas, corre rapidamente por um campo ao ar livre.

Dados visuais:
{}

Audiodescrição:
""".format(input_json).strip()