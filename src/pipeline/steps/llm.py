# src/pipeline/steps/llm.py
from .base_step import BaseStep
from src.openai_client import get_async_openai_client

class LLMGenerationStep(BaseStep):
    """
    Шаг для генерации текста с помощью LLM.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.input_key = self.config['input_key']
        self.output_key = self.config['output_key']
        
        # Инициализация клиента OpenAI для этого шага
        client_config = self.config['client_config']
        self.model_name = self.config['model_name']
        self.params = self.config.get('params', {})
        self.client = get_async_openai_client(client_config)

    async def process(self, item: dict) -> dict:
        """
        Берет промпт из item, отправляет в LLM и добавляет ответ в item.
        """
        prompt_text = item.get(self.input_key)
        
        if not prompt_text:
            self.logger.warning(f"В записи отсутствует ключ '{self.input_key}' или он пуст. Пропуск генерации.")
            item[self.output_key] = None
            return item
            
        try:
            # Используем messages-формат для совместимости с ChatCompletion
            messages = [{"role": "user", "content": prompt_text}]
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **self.params
            )
            llm_response_text = response.choices[0].message.content
            item[self.output_key] = llm_response_text
        except Exception as e:
            self.logger.error(f"Ошибка при вызове LLM API для модели {self.model_name}: {e}", exc_info=True)
            item[self.output_key] = None
            
        return item
