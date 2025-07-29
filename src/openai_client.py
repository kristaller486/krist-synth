# Модуль для взаимодействия с OpenAI API
import os
import logging
from openai import OpenAI, AsyncOpenAI, RateLimitError, APIError, APITimeoutError
import time
import asyncio # Добавлено для асинхронных операций
import httpx # Добавлено для настройки SSL
from typing import NamedTuple, Optional

class GenerationResult(NamedTuple):
    text: Optional[str]
    reasoning_content: Optional[str]


def get_openai_client(client_config: dict) -> OpenAI:
    api_key = client_config.get('api_key') or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Ключ OpenAI API не найден. Укажите его в секции 'openai_client' конфига или переменной окружения OPENAI_API_KEY.")

    base_url = client_config.get('base_url')
    timeout = client_config.get('timeout')
    max_retries = client_config.get('max_retries')

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            # for local models
            http_client=httpx.Client(verify=False)
        )

        logging.info(f"Синхронный клиент OpenAI инициализирован с отключенной проверкой SSL (base_url: {base_url or 'default'}, timeout: {timeout or 'default'}, max_retries: {max_retries or 'default'}).")
        return client
    except Exception as e:
        logging.error(f"Ошибка инициализации синхронного клиента OpenAI: {e}", exc_info=True)
        raise

def get_async_openai_client(client_config: dict) -> AsyncOpenAI:
    """
    Инициализирует и возвращает асинхронный клиент OpenAI на основе конфигурации.
    :param client_config: Словарь с настройками клиента ('api_key', 'base_url', 'timeout', 'max_retries').
    """
    api_key = client_config.get('api_key') or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Ключ OpenAI API не найден. Укажите его в секции 'openai_client' конфига или переменной окружения OPENAI_API_KEY.")

    base_url = client_config.get('base_url')
    timeout = client_config.get('timeout')
    max_retries = client_config.get('max_retries')

    try:
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            # Передаем настроенный httpx клиент для отключения проверки SSL
            http_client=httpx.AsyncClient(verify=False)
        )
        logging.info(f"Асинхронный клиент OpenAI инициализирован с отключенной проверкой SSL (base_url: {base_url or 'default'}, timeout: {timeout or 'default'}, max_retries: {max_retries or 'default'}).")
        return client
    except Exception as e:
        logging.error(f"Ошибка инициализации асинхронного клиента OpenAI: {e}", exc_info=True)
        raise

def generate_text(client: OpenAI, model: str, prompt: str, params: dict, retries: int = 3, delay: int = 5) -> str | None:
    """Генерирует текст с использованием указанной модели OpenAI и параметров с логикой повторных попыток."""
    logging.debug(f"Генерация текста с моделью '{model}' с использованием промпта (первые 100 символов): {prompt[:100]}...")
    attempt = 0
    while attempt < retries:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                
                **params
            )
            # Предполагаем, что нам нужно содержимое первого варианта
            generated_text = response.choices[0].message.content.strip()
            logging.debug(f"Текст успешно сгенерирован (первые 100 символов): {generated_text[:100]}...")
            return generated_text
        except RateLimitError as e:
            attempt += 1
            logging.warning(f"Превышен лимит запросов. Повторная попытка {attempt}/{retries} через {delay} секунд... Ошибка: {e}")
            time.sleep(delay)
            delay *= 2 # Экспоненциальная задержка
        except (APIError, APITimeoutError) as e:
            attempt += 1
            logging.warning(f"Произошла ошибка API или тайм-аут. Повторная попытка {attempt}/{retries} через {delay} секунд... Ошибка: {e}")
            time.sleep(delay)
        except Exception as e:
            logging.error(f"Произошла непредвиденная ошибка во время вызова OpenAI API: {e}")
            # В зависимости от ошибки, повторная попытка может не иметь смысла
            return None # Или перевыбросить исключение

    logging.error(f"Не удалось сгенерировать текст после {retries} попыток для промпта: {prompt[:100]}...")
    return None


async def generate_text_async(client: AsyncOpenAI, model: str, prompt: str, params: dict, retries: int = 3, delay: int = 5) -> GenerationResult | None:
    """
    Асинхронно генерирует текст и reasoning_content (если доступно) 
    с использованием указанной модели OpenAI и параметров с логикой повторных попыток.
    Возвращает объект GenerationResult или None в случае полной неудачи.
    """
    logging.debug(f"Асинхронная генерация текста с моделью '{model}' для промпта (первые 100 символов): {prompt[:100]}...")
    attempt = 0
    current_delay = delay
    while attempt < retries:
        try:
            # Запрашиваем reasoning_content, если это поддерживается моделью/API
            # Это может потребовать специфичного параметра в params, например, include_reasoning=True
            # или это может быть стандартным полем в ответе для некоторых моделей/эндпоинтов.
            # OpenAI API обычно не возвращает 'reasoning_content' в стандартном ChatCompletion.
            # Если ваш API endpoint возвращает это, код ниже должен работать.
            # В противном случае, reasoning_content будет None.
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **params  # Убедитесь, что params могут включать запрос на reasoning, если это необходимо
            )
            
            generated_text = None
            reasoning_content = None

            if response.choices and response.choices[0].message:
                if response.choices[0].message.content:
                    generated_text = response.choices[0].message.content.strip()
                
                # Пытаемся извлечь reasoning_content, если он есть
                # Это нестандартное поле для OpenAI, может быть специфичным для вашего base_url
                if hasattr(response.choices[0].message, 'reasoning_content') and response.choices[0].message.reasoning_content:
                    reasoning_content = response.choices[0].message.reasoning_content
                elif hasattr(response, 'reasoning_content') and response.reasoning_content: # Иногда может быть на верхнем уровне
                     reasoning_content = response.reasoning_content


            if generated_text is not None:
                 logging.debug(f"Текст успешно сгенерирован асинхронно (первые 100 символов): {generated_text[:100]}...")
            else:
                 logging.warning(f"Ответ от API получен, но не содержит текста (model: {model}, prompt: {prompt[:50]}...).")
            
            return GenerationResult(text=generated_text, reasoning_content=reasoning_content)

        except RateLimitError as e:
            attempt += 1
            logging.warning(f"[Async] Превышен лимит запросов. Повторная попытка {attempt}/{retries} через {current_delay} секунд... Ошибка: {e}")
            await asyncio.sleep(current_delay)
            current_delay *= 2 # Экспоненциальная задержка
        except (APIError, APITimeoutError) as e:
            attempt += 1
            # Добавляем exc_info=True для вывода трейсбека
            logging.warning(f"[Async] Произошла ошибка API или тайм-аут. Повторная попытка {attempt}/{retries} через {current_delay} секунд... Ошибка: {e}", exc_info=True)
            await asyncio.sleep(current_delay)
        except Exception as e:
            # Логируем конкретный промпт при ошибке
            logging.error(f"[Async] Произошла непредвиденная ошибка во время вызова OpenAI API для промпта '{prompt[:100]}...': {e}", exc_info=True)
            # В зависимости от ошибки, повторная попытка может не иметь смысла
            return GenerationResult(text=None, reasoning_content=None) # Возвращаем объект с None значениями

    logging.error(f"[Async] Не удалось сгенерировать текст после {retries} попыток для промпта: {prompt[:100]}...")
    return GenerationResult(text=None, reasoning_content=None) # Возвращаем объект с None значениями
