# src/pipeline/steps/multi_turn_generation.py
# pylint: disable=too-many-locals, too-many-branches
"""
Multi-turn data regeneration step.

Принимает колонку с диалогом в формате ShareGPT и
перезаписывает все реплики с `from == "gpt"` новыми ответами,
сгенерированными выбранной LLM-моделью.

Config-schema
-------------
input_key: str                 # обязательный, имя поля с диалогом
model_name: str                # обязательный, имя модели для OpenAI-совместимого API
client_config: dict            # обязательный, параметры клиента (api_key/base_url…)
params: dict                   # необязательный, доп. параметры chat.completions
model_name_key: str | None     # необязательный, куда записать имя модели
"""
from __future__ import annotations

from typing import List, Dict, Any

from .base_step import BaseStep
from src.openai_client import get_async_openai_client


ROLE_MAP = {
    "system": "system",
    "user": "user",
    "human": "user",
    "gpt": "assistant",
    "assistant": "assistant",
}


class MultiTurnGenerationStep(BaseStep):
    """
    Шаг, который заново генерирует все GPT-реплики в диалоге.

    • `input_key`  — столбец с диалогом `List[Dict[from,value]]`
    • `model_name` — имя модели для chat-completion
    • `model_name_key` (optional) — куда положить имя модели в item
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.input_key: str = self.config["input_key"]
        self.model_name: str = self.config["model_name"]
        self.model_name_key: str | None = self.config.get("model_name_key")

        self.params: dict[str, Any] = self.config.get("params", {})
        self.client = get_async_openai_client(self.config["client_config"])

    async def _chat_completion(self, messages: List[Dict[str, str]]) -> str | None:
        """
        Отдельная корутина для вызова LLM, чтобы облегчить чтение.
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **self.params,
            )
            return response.choices[0].message.content
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.error("Multi-turn generation error: %s", exc, exc_info=True)
            return None

    async def process(self, item: dict, batch_id: int = 0) -> dict:  # noqa: D401, N802
        """
        Main step logic.

        1. Читаем список `turns` из input_key.
        2. Итерируемся, копируя system/user как есть.
        3. Когда встречаем `gpt`, вызываем LLM на текущем контексте
           и подменяем реплику новым ответом.
        4. Записываем результат обратно в тот же столбец.
        """
        turns: list[dict] = item.get(self.input_key)  # type: ignore
        if not turns:
            self.logger.warning(
                "Запись не содержит ключ '%s' или он пуст. batch=%s", self.input_key, batch_id
            )
            return item

        new_dialog: list[dict] = []
        messages: list[dict[str, str]] = []

        for turn in turns:
            role = turn.get("from")
            value = turn.get("value", "")

            if role != "gpt":
                # Копируем как есть и добавляем в контекст
                new_dialog.append(turn)
                mapped_role = ROLE_MAP.get(role, "user")
                messages.append({"role": mapped_role, "content": value})
                continue

            # Встречена GPT-реплика → генерируем заново
            assistant_response = await self._chat_completion(messages)

            # Если не удалось сгенерировать — пишем None, чтобы отследить
            new_turn = {"from": "gpt", "value": assistant_response}
            new_dialog.append(new_turn)

            # Добавляем ответ в контекст, даже если None → пустая строка
            messages.append({"role": "assistant", "content": assistant_response or ""})

        # Перезаписываем оригинальное поле
        item[self.input_key] = new_dialog

        # При необходимости сохраняем имя модели
        if self.model_name_key:
            item[self.model_name_key] = self.model_name

        return item
