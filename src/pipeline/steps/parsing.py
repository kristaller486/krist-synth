# src/pipeline/steps/parsing.py
import json
import re
from .base_step import BaseStep

class ExtractJSONStep(BaseStep):
    """
    Шаг для извлечения JSON объекта из текстового поля.
    Ищет первый валидный JSON между ```json ... ``` или просто в тексте.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.input_key = self.config['input_key']
        self.output_key = self.config['output_key']
        # Если true, распаковывает JSON в `item`, иначе кладет сам JSON-объект.
        self.unpack = self.config.get('unpack', True)

    async def process(self, item: dict) -> dict:
        """
        Извлекает JSON из текстового поля и добавляет его в item.
        """
        text_content = item.get(self.input_key)
        
        if not text_content:
            self.logger.warning(f"Поле '{self.input_key}' пустое. Пропуск извлечения JSON.")
            if not self.unpack:
                item[self.output_key] = None
            return item

        json_data = None
        try:
            # Сначала ищем JSON в блоках ```json ... ```
            match = re.search(r'```json\s*(\{.*?\})\s*```', text_content, re.DOTALL)
            if match:
                json_str = match.group(1)
                json_data = json.loads(json_str)
            else:
                # Если не нашли, ищем первый попавшийся JSON-объект в тексте
                match = re.search(r'(\{.*?\})', text_content, re.DOTALL)
                if match:
                    json_str = match.group(1)
                    json_data = json.loads(json_str)

            if json_data:
                if self.unpack:
                    # Распаковываем ключи из JSON в основной словарь item
                    item.update(json_data)
                else:
                    # Кладем весь объект в указанный ключ
                    item[self.output_key] = json_data
            else:
                self.logger.warning(f"Не удалось найти валидный JSON в поле '{self.input_key}'.")
                if not self.unpack:
                    item[self.output_key] = None

        except json.JSONDecodeError as e:
            self.logger.error(f"Ошибка декодирования JSON из поля '{self.input_key}': {e}")
            if not self.unpack:
                item[self.output_key] = None
        except Exception as e:
            self.logger.error(f"Непредвиденная ошибка при извлечении JSON: {e}")
            if not self.unpack:
                item[self.output_key] = None

        return item
