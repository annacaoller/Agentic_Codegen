from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


class LLMError(RuntimeError):
    pass


@dataclass
class OpenAILLM:
    model: str = "gpt-5"  # override via OPENAI_MODEL
    temperature: float = 0.0

    def __post_init__(self) -> None:
        env_model = os.getenv("OPENAI_MODEL")
        if env_model:
            self.model = env_model

    def generate(self, prompt: str) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMError("OPENAI_API_KEY is not set in environment.")

        # Preferred (new SDK):
        try:
            from openai import OpenAI  # type: ignore
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": "You are a reliable coding agent."},
                    {"role": "user", "content": prompt},
                ],
            )
            return resp.choices[0].message.content or ""
        except Exception:
            # Fallback (older SDK):
            try:
                import openai  # type: ignore

                openai.api_key = api_key
                resp = openai.ChatCompletion.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": "You are a reliable coding agent."},
                        {"role": "user", "content": prompt},
                    ],
                )
                return resp["choices"][0]["message"]["content"] or ""
            except Exception as e:
                raise LLMError(f"OpenAI call failed: {type(e).__name__}: {e}") from e
