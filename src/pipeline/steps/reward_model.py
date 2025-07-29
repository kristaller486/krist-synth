# src/pipeline/steps/reward_model.py
"""
Reward Model Approval step.

Принимает preference данные (prompt, chosen, rejected) и использует
reward модель для оценки качества ответов и определения аппрува.

Config-schema:
input_prompt_key: str          # обязательный, поле с промптом
input_chosen_key: str          # обязательный, поле с chosen ответом  
input_rejected_key: str        # обязательный, поле с rejected ответом
model_name: str                # обязательный, имя RM модели
client_config: dict            # обязательный, параметры клиента
params: dict                   # необязательный, доп. параметры API
"""
from __future__ import annotations

import re
import requests
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer

from .base_step import BaseStep
from src.openai_client import get_async_openai_client


class RewardModelApprovalStep(BaseStep):
    """
    Шаг для аппрува preference данных с помощью reward модели.
    
    Формирует messages из prompt + chosen/rejected и отправляет в RM.
    Парсит ответ для получения scores и определения approval статуса.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.input_prompt_key: str = self.config["input_prompt_key"]
        self.input_chosen_key: str = self.config["input_chosen_key"] 
        self.input_rejected_key: str = self.config["input_rejected_key"]
        self.model_name: str = self.config["model_name"]
        self.params: dict = self.config.get("params", {})
        self.client = get_async_openai_client(self.config["client_config"])

    def _build_messages(self, prompt: List[Dict], chosen: List[Dict], rejected: List[Dict]) -> List[Dict[str, str]]:
        """
        Формирует messages для reward модели из prompt, chosen, rejected.
        
        Формат: prompt_messages + response_1 (chosen) + response_2 (rejected)
        """
        messages = []
        
        # Добавляем prompt messages
        for msg in prompt:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role and content:
                messages.append({"role": role, "content": content})
        
        # Добавляем chosen как response_1
        for msg in chosen:
            content = msg.get("content", "")
            if content:
                messages.append({"role": "response_1", "content": content})
                break  # берем только первое сообщение
        
        # Добавляем rejected как response_2  
        for msg in rejected:
            content = msg.get("content", "")
            if content:
                messages.append({"role": "response_2", "content": content})
                break  # берем только первое сообщение
                
        return messages

    async def _reward_model_completion(self, messages: List[Dict[str, str]]) -> str | None:
        """
        Вызов reward модели для оценки двух ответов.
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **self.params,
            )
            return response.choices[0].message.content
        except Exception as exc:
            self.logger.error("Reward model API error: %s", exc, exc_info=True)
            return None

    def _parse_rm_output(self, raw_output: str) -> Tuple[int | None, int | None, int | None, str]:
        """
        Парсит ответ reward модели для извлечения scores.
        
        Returns:
            (chosen_score, rejected_score, ranking_score, filtered_output)
        """
        if not raw_output:
            return None, None, None, ""
            
        # Убираем think часть
        filtered_output = raw_output.split("</think>")[-1].strip()
        
        chosen_score = None
        rejected_score = None  
        ranking_score = None
        
        try:
            # Ищем Individual Scores: \boxed{score1, score2} или \boxed{score}
            individual_match = re.search(r'\\boxed\{([^}]+)\}', filtered_output)
            if individual_match:
                scores_text = individual_match.group(1).strip()
                if ',' in scores_text:
                    # Два score'а
                    parts = [s.strip() for s in scores_text.split(',')]
                    if len(parts) >= 2:
                        chosen_score = int(parts[0])
                        rejected_score = int(parts[1])
                else:
                    # Один score (когда оценивается только один ответ)
                    chosen_score = int(scores_text)
            
            # Ищем Ranking Score (обычно последний \boxed)
            ranking_matches = re.findall(r'\\boxed\{([^}]+)\}', filtered_output)
            if len(ranking_matches) >= 2:
                # Если есть два boxed, второй это ranking score
                ranking_score = int(ranking_matches[1].strip())
            elif len(ranking_matches) == 1 and chosen_score is None:
                # Если только один boxed и нет individual scores, это может быть ranking
                ranking_score = int(ranking_matches[0].strip())
                
        except (ValueError, IndexError) as e:
            self.logger.warning("Ошибка парсинга RM scores: %s", e)
            
        return chosen_score, rejected_score, ranking_score, filtered_output

    async def process(self, item: dict, batch_id: int = 0) -> dict:
        """
        Основная логика шага.
        
        1. Извлекает prompt, chosen, rejected из item
        2. Формирует messages для RM
        3. Вызывает RM API 
        4. Парсит результат и добавляет поля в item
        """
        prompt = item.get(self.input_prompt_key)
        chosen = item.get(self.input_chosen_key)
        rejected = item.get(self.input_rejected_key)
        
        # Валидация входных данных
        if not prompt:
            self.logger.warning("Поле '%s' пустое. batch=%s", self.input_prompt_key, batch_id)
            return self._add_default_fields(item)
            
        if not chosen:
            self.logger.warning("Поле '%s' пустое. batch=%s", self.input_chosen_key, batch_id)
            return self._add_default_fields(item)
            
        if not rejected:
            self.logger.warning("Поле '%s' пустое. batch=%s", self.input_rejected_key, batch_id)
            return self._add_default_fields(item)
        
        # Формируем messages
        try:
            messages = self._build_messages(prompt, chosen, rejected)
            if not messages:
                self.logger.warning("Не удалось сформировать messages. batch=%s", batch_id)
                return self._add_default_fields(item)
        except Exception as e:
            self.logger.error("Ошибка формирования messages: %s batch=%s", e, batch_id)
            return self._add_default_fields(item)
        
        # Вызываем RM
        raw_output = await self._reward_model_completion(messages)
        if not raw_output:
            self.logger.warning("RM не вернула результат. batch=%s", batch_id)
            return self._add_default_fields(item, raw_output="")
        
        # Парсим результат
        chosen_score, rejected_score, ranking_score, filtered_output = self._parse_rm_output(raw_output)
        
        # Определяем approval
        rm_approved = False
        if ranking_score is not None:
            # ranking_score 1-3: chosen лучше rejected
            rm_approved = ranking_score in [1, 2, 3]
        
        # Добавляем поля в item
        item["rm_approved"] = rm_approved
        item["ranking_score"] = ranking_score
        item["chosen_individual_score"] = chosen_score  
        item["rejected_individual_score"] = rejected_score
        item["rm_raw_output"] = raw_output
        item["rm_output"] = filtered_output
        
        return item

    def _add_default_fields(self, item: dict, raw_output: str = None) -> dict:
        """Добавляет поля со значениями по умолчанию при ошибке."""
        item["rm_approved"] = False
        item["ranking_score"] = None
        item["chosen_individual_score"] = None
        item["rejected_individual_score"] = None  
        item["rm_raw_output"] = raw_output
        item["rm_output"] = None
        return item


class SkyworkRewardModelApprovalStep(BaseStep):
    """
    Шаг для аппрува preference данных с помощью Skywork Reward Model.
    
    Использует HTTP API для получения числовых оценок вместо текстового анализа.
    Модель возвращает float scores, где большее значение = лучший ответ.
    
    Config-schema:
    base_url: str                  # http://127.0.0.1:8000/classify
    model_name: str                # Skywork/Skywork-Reward-V2-Llama-3.1-8B
    input_prompt_key: str          # поле с промптом
    input_chosen_key: str          # поле с chosen ответом
    input_rejected_key: str        # поле с rejected ответом
    timeout: int                   # таймаут HTTP запроса (опционально)
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.base_url: str = self.config["base_url"]
        self.model_name: str = self.config["model_name"]
        self.input_prompt_key: str = self.config["input_prompt_key"]
        self.input_chosen_key: str = self.config["input_chosen_key"]
        self.input_rejected_key: str = self.config["input_rejected_key"]
        self.timeout: int = self.config.get("timeout", 60)
        
        # Инициализируем токенизатор
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except Exception as e:
            self.logger.error("Ошибка загрузки токенизатора %s: %s", self.model_name, e)
            raise

    def _build_conversation(self, prompt: List[Dict], response: List[Dict]) -> List[Dict]:
        """Формирует диалог из prompt + response для токенизации."""
        conversation = []
        
        # Добавляем все сообщения из prompt
        for msg in prompt:
            role = msg.get("role")
            content = msg.get("content", "")
            if role and content:
                conversation.append({"role": role, "content": content})
        
        # Добавляем ответ (берем первое сообщение)
        for msg in response:
            content = msg.get("content", "")
            if content:
                conversation.append({"role": "assistant", "content": content})
                break
                
        return conversation

    def _format_conversations(self, conversations: List[List[Dict]]) -> List[str]:
        """
        Форматирует диалоги через tokenizer.apply_chat_template.
        Убирает BOS токен если присутствует.
        """
        formatted_convs = []
        for conv in conversations:
            try:
                formatted = self.tokenizer.apply_chat_template(conv, tokenize=False)
                # Убираем BOS токен если есть
                if self.tokenizer.bos_token is not None and formatted.startswith(self.tokenizer.bos_token):
                    formatted = formatted[len(self.tokenizer.bos_token):]
                formatted_convs.append(formatted)
            except Exception as e:
                self.logger.error("Ошибка форматирования диалога: %s", e)
                formatted_convs.append("")  # fallback
                
        return formatted_convs

    def _call_skywork_api(self, conversations: List[List[Dict]]) -> List[float | None]:
        """
        Вызывает Skywork API для получения scores.
        
        Returns:
            List[float | None]: Список scores для каждого диалога
        """
        try:
            # Форматируем диалоги
            formatted_convs = self._format_conversations(conversations)
            
            # Подготавливаем payload
            payload = {
                "model": self.model_name,
                "text": formatted_convs
            }
            
            # Отправляем HTTP запрос
            response = requests.post(
                self.base_url, 
                json=payload, 
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Парсим ответ
            result = response.json()
            scores = []
            
            for item in result:
                if "embedding" in item and len(item["embedding"]) > 0:
                    scores.append(float(item["embedding"][0]))
                else:
                    scores.append(None)
                    
            if len(scores) != len(conversations):
                self.logger.warning("Получено %d scores для %d диалогов", len(scores), len(conversations))
                
            return scores
            
        except requests.RequestException as e:
            self.logger.error("HTTP ошибка при вызове Skywork API: %s", e)
            return [None] * len(conversations)
        except Exception as e:
            self.logger.error("Ошибка при вызове Skywork API: %s", e)
            return [None] * len(conversations)

    async def process(self, item: dict, batch_id: int = 0) -> dict:
        """
        Основная логика шага.
        
        1. Извлекает prompt, chosen, rejected из item
        2. Формирует диалоги для каждого ответа
        3. Вызывает Skywork API для получения scores
        4. Сравнивает scores и определяет approval
        """
        prompt = item.get(self.input_prompt_key)
        chosen = item.get(self.input_chosen_key)
        rejected = item.get(self.input_rejected_key)
        
        # Валидация входных данных
        if not prompt:
            self.logger.warning("Поле '%s' пустое. batch=%s", self.input_prompt_key, batch_id)
            return self._add_skywork_default_fields(item)
            
        if not chosen:
            self.logger.warning("Поле '%s' пустое. batch=%s", self.input_chosen_key, batch_id)
            return self._add_skywork_default_fields(item)
            
        if not rejected:
            self.logger.warning("Поле '%s' пустое. batch=%s", self.input_rejected_key, batch_id)
            return self._add_skywork_default_fields(item)
        
        # Формируем диалоги
        try:
            chosen_conv = self._build_conversation(prompt, chosen)
            rejected_conv = self._build_conversation(prompt, rejected)
            
            if not chosen_conv or not rejected_conv:
                self.logger.warning("Не удалось сформировать диалоги. batch=%s", batch_id)
                return self._add_skywork_default_fields(item)
                
        except Exception as e:
            self.logger.error("Ошибка формирования диалогов: %s batch=%s", e, batch_id)
            return self._add_skywork_default_fields(item)
        
        # Вызываем Skywork API
        scores = self._call_skywork_api([chosen_conv, rejected_conv])
        chosen_score, rejected_score = scores[0], scores[1]
        
        # Определяем approval
        rm_approved = False
        score_difference = None
        
        if chosen_score is not None and rejected_score is not None:
            rm_approved = chosen_score > rejected_score
            score_difference = chosen_score - rejected_score
        
        # Добавляем поля в item
        item["rm_approved"] = rm_approved
        item["chosen_score"] = chosen_score
        item["rejected_score"] = rejected_score
        item["score_difference"] = score_difference
        
        return item

    def _add_skywork_default_fields(self, item: dict) -> dict:
        """Добавляет поля со значениями по умолчанию при ошибке."""
        item["rm_approved"] = False
        item["chosen_score"] = None
        item["rejected_score"] = None
        item["score_difference"] = None
        return item
