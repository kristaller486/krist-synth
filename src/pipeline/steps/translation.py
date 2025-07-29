# src/pipeline/steps/translation.py
from .base_step import BaseStep
from .prompt import FormatPromptStep
from .llm import LLMGenerationStep
import copy
import logging
import asyncio

logger = logging.getLogger(__name__)

class IterativeTranslationStep(BaseStep):
    """
    Шаг для итеративного перевода диалога с кэшированием системных сообщений.
    """
    system_prompt_cache = {}
    _cache_lock: asyncio.Lock = asyncio.Lock()
    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # --- Настройка для системных сообщений (с кэшем) ---
        if 'system_translation_config' in self.config:
            sys_config = self.config['system_translation_config']
            self.sys_formatter = FormatPromptStep(sys_config['formatter'])
            self.sys_llm_generator = LLMGenerationStep(sys_config['llm'])
        else:
            self.sys_formatter = None
            self.sys_llm_generator = None

        # --- Настройка для обычных реплик ---
        conv_config = self.config['conversation_translation_config']
        self.conv_formatter = FormatPromptStep(conv_config['formatter'])
        self.conv_llm_generator = LLMGenerationStep(conv_config['llm'])
        
        # --- Ключи для работы с данными ---
        self.source_conversations_key = self.config['source_conversations_key']
        self.target_conversations_key = self.config['target_conversations_key']
        self.backup_conversations_key = self.config['backup_conversations_key']

    async def _translate_system_prompt(self, turn: dict) -> str | None:
        """
        Переводит системный промпт с высокопроизводительным кэшем. Это всё магия от o3.

        Алгоритм:
        1. Быстрый неблокирующий поиск строки в кэше.
           • Если там готовый `str` — hit.
           • Если там `Future` — дожидаемся его результата.
        2. На промахе создаём `Future` и помещаем в кэш под короткой блокировкой.
           Остальные корутины будут ждать этот `Future` вместо повторного перевода.
        3. Сам перевод выполняем без блокировки. По завершении пишем
           результат в `Future` и заменяем значение в кэше на готовую строку.
        """
        system_text: str | None = turn.get("value")
        if not system_text:
            return None

        # --- 1. Быстрый путь без блокировки ---
        cached = self.system_prompt_cache.get(system_text)
        if isinstance(cached, str):
            self.logger.debug(f"Cache hit! Prompt: {system_text[:30]}")
            return cached
        if isinstance(cached, asyncio.Future):
            result = await cached
            self.logger.debug(f"Cache hit(wait)! Prompt: {system_text[:30]}")
            return result

        # --- 2. Создаём Future и кладём в кэш под блокировкой ---
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str | None] = loop.create_future()
        async with self._cache_lock:
            # double-check, пока ждали lock
            cached = self.system_prompt_cache.get(system_text)
            if isinstance(cached, str):
                future.cancel()
                self.logger.debug(f"Cache hit(dc)! Prompt: {system_text[:30]}")
                return cached
            if isinstance(cached, asyncio.Future):
                future.cancel()
                result = await cached
                self.logger.debug(f"Cache hit(wait-dc)! Prompt: {system_text[:30]}")
                return result
            # ставим «заглушку», чтобы остальные ждали
            self.system_prompt_cache[system_text] = future

        translated_text: str | None = None

        # --- 3. Собственно перевод (без блокировки) ---
        if self.sys_formatter and self.sys_llm_generator:
            ctx = {"text": system_text}
            ctx = await self.sys_formatter.process(ctx)
            ctx = await self.sys_llm_generator.process(ctx)
            translated_text = ctx.get(self.sys_llm_generator.output_key)

        # --- 4. Завершаем Future и финализируем кэш ---
        future.set_result(translated_text)

        async with self._cache_lock:
            if translated_text:
                self.system_prompt_cache[system_text] = translated_text
                self.logger.debug(f"Write cache! Prompt: {system_text[:30]}")
            else:
                # Удаляем пустой результат, чтобы попробовать снова позже
                self.system_prompt_cache.pop(system_text, None)

        return translated_text

    async def process(self, item: dict) -> dict:
        original_conversations = item.get(self.source_conversations_key, [])
        if not original_conversations:
            return item
            
        item[self.backup_conversations_key] = copy.deepcopy(original_conversations)
        
        final_conversations = []
        context_history = []
        
        for i, turn in enumerate(original_conversations):
            role = turn.get('from')
            
            if role == 'system' and self.sys_llm_generator:
                translated_text = await self._translate_system_prompt(turn)
                if translated_text:
                    translated_turn = {'from': 'system', 'value': translated_text.strip()}
                    final_conversations.append(translated_turn)
                    context_history.append(translated_turn)
                else:
                    logger.error(f"Не удалось перевести системное сообщение. Пропускаем.")
                    final_conversations.append(turn) # Добавляем оригинал в случае ошибки
                    context_history.append(turn)

            elif role == 'human':
                processing_context = {'dialog_history': context_history, 'current_utterance': turn}
                context = await self.conv_formatter.process(processing_context)
                context = await self.conv_llm_generator.process(context)
                translated_text = context.get(self.conv_llm_generator.output_key)
                
                if translated_text:
                    translated_turn = {'from': 'human', 'value': translated_text.strip()}
                    final_conversations.append(translated_turn)
                    context_history.append(translated_turn)
                else:
                    logger.error(f"Не удалось перевести 'human' реплику {i}. Прерываем для записи.")
                    return item
            
            elif role == 'gpt':
                final_conversations.append(turn)
                context_history.append(turn)

        item[self.target_conversations_key] = final_conversations
        return item
