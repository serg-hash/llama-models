# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

"""Agente base para Moltbook.

Este módulo define un agente configurable que organiza notas, genera resúmenes
breves y propone acciones para un cuaderno digital llamado Moltbook.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from jinja2 import Template

DEFAULT_SYSTEM_PROMPT = """
Eres Moltbook, un asistente de IA para organizar cuadernos digitales.
Tu misión es ayudar a las personas a capturar, estructurar y recuperar ideas.
Responde siempre en español claro y directo.

Reglas:
- Resume con precisión sin inventar datos.
- Propón etiquetas útiles y accionables.
- Sugiere próximos pasos concretos.
- Cuando falte información, formula preguntas breves.
""".strip()

PROMPT_TEMPLATE = Template(
    """
{{ system_prompt }}

Contexto de usuario:
{{ context or "(sin contexto adicional)" }}

Nota actual:
{{ note or "(sin nota adjunta)" }}

Solicitud del usuario:
{{ user_message }}

Responde con:
1. Resumen (máx. {{ max_summary_tokens }} palabras)
2. Etiquetas sugeridas ({{ max_tags }} máximo)
3. Acciones recomendadas ({{ max_action_items }} máximo)
4. Preguntas de aclaración si son necesarias
""".strip()
)


@dataclass
class ToolSpec:
    name: str
    description: str
    arguments: Dict[str, str]


@dataclass
class MoltbookAgentConfig:
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    locale: str = "es-ES"
    max_summary_tokens: int = 120
    max_tags: int = 6
    max_action_items: int = 5
    tools: List[ToolSpec] = field(
        default_factory=lambda: [
            ToolSpec(
                name="buscar_en_moltbook",
                description="Busca notas relacionadas por texto o etiqueta.",
                arguments={"query": "Texto o etiquetas a buscar."},
            ),
            ToolSpec(
                name="crear_recordatorio",
                description="Crea un recordatorio con fecha y contexto.",
                arguments={"titulo": "Título breve.", "fecha": "ISO-8601"},
            ),
        ]
    )


class MoltbookAgent:
    def __init__(self, config: Optional[MoltbookAgentConfig] = None) -> None:
        self.config = config or MoltbookAgentConfig()

    def build_prompt(
        self,
        user_message: str,
        note: Optional[str] = None,
        context: Optional[str] = None,
    ) -> str:
        return PROMPT_TEMPLATE.render(
            system_prompt=self.config.system_prompt,
            user_message=user_message,
            note=note,
            context=context,
            max_summary_tokens=self.config.max_summary_tokens,
            max_tags=self.config.max_tags,
            max_action_items=self.config.max_action_items,
        )

    def generate_response(
        self,
        llm: Callable[[str], str],
        user_message: str,
        note: Optional[str] = None,
        context: Optional[str] = None,
    ) -> str:
        prompt = self.build_prompt(user_message, note=note, context=context)
        return llm(prompt)

    def analyze_note(
        self,
        llm: Callable[[str], str],
        note: str,
        context: Optional[str] = None,
    ) -> str:
        user_message = (
            "Analiza la nota para crear un resumen, etiquetas y acciones."
        )
        return self.generate_response(llm, user_message, note=note, context=context)


def demo_llm(prompt: str) -> str:
    return f"--- PROMPT GENERADO ---\n{prompt}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Agente base para Moltbook.")
    parser.add_argument(
        "--nota",
        dest="note",
        help="Contenido de la nota a analizar.",
    )
    parser.add_argument(
        "--contexto",
        dest="context",
        help="Contexto adicional del usuario.",
    )
    parser.add_argument(
        "--mensaje",
        dest="message",
        default="Ayúdame a organizar esta nota.",
        help="Solicitud del usuario.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Imprime el prompt generado en lugar de llamar a un modelo.",
    )
    args = parser.parse_args()

    agent = MoltbookAgent()
    if args.demo:
        response = agent.generate_response(
            demo_llm, args.message, note=args.note, context=args.context
        )
    else:
        response = agent.generate_response(
            demo_llm, args.message, note=args.note, context=args.context
        )
    print(response)


if __name__ == "__main__":
    main()
