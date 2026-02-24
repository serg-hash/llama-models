"""CLI agent specialized for OpenClaw assistance.

This script creates a small conversational agent on top of any
instruction-tuned model available through Hugging Face Transformers.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import List


SYSTEM_PROMPT = (
    "Eres OpenClaw Agent, un asistente técnico enfocado en el juego OpenClaw. "
    "Ayudas con instalación, compilación, configuración, mods, depuración y "
    "automatización. Responde en español salvo que el usuario pida otro idioma. "
    "Da pasos concretos, comandos listos para copiar y buenas prácticas."
)


@dataclass
class Message:
    role: str
    content: str


@dataclass
class OpenClawAgent:
    model_id: str
    temperature: float = 0.2
    max_new_tokens: int = 512
    history: List[Message] = field(default_factory=list)

    def __post_init__(self) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "No se encontró transformers. Instala dependencias con: "
                "pip install transformers torch"
            ) from exc

        self._pipeline = pipeline(
            "text-generation",
            model=self.model_id,
            device_map="auto",
        )
        self.history.append(Message(role="system", content=SYSTEM_PROMPT))

    def reply(self, user_message: str) -> str:
        self.history.append(Message(role="user", content=user_message))
        conversation = "\n".join(f"{m.role}: {m.content}" for m in self.history)

        result = self._pipeline(
            conversation + "\nassistant:",
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            return_full_text=False,
        )
        answer = result[0]["generated_text"].strip()
        self.history.append(Message(role="assistant", content=answer))
        return answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecuta un agente especializado en OpenClaw")
    parser.add_argument(
        "--model-id",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Modelo de Hugging Face a usar",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = OpenClawAgent(
        model_id=args.model_id,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    print("OpenClaw Agent listo. Escribe tu consulta ('salir' para terminar).")
    while True:
        prompt = input("Tú: ").strip()
        if prompt.lower() in {"salir", "exit", "quit"}:
            print("Hasta luego.")
            break

        response = agent.reply(prompt)
        print(f"Agente: {response}\n")


if __name__ == "__main__":
    main()
