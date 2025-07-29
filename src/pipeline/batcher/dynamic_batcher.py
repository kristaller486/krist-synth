# src/pipeline/batcher/dynamic_batcher.py
import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, List

from datasets import Dataset
from tqdm.auto import tqdm

from src.pipeline.batcher.base_batcher import BaseBatcher

logger = logging.getLogger(__name__)


class DynamicBatcher(BaseBatcher):
    def __init__(self, steps: List[Any], config: Dict[str, Any]):
        super().__init__(steps, config)
        self.max_concurrent_tasks = self.config.get('max_concurrent_tasks', 10)

    async def process(self, ds: Dataset) -> AsyncGenerator[Dict[str, Any], None]:
        total_items = len(ds)
        pbar = tqdm(total=total_items, desc="Динамическая обработка")

        async def item_producer(items):
            for item in items:
                yield item

        async def worker(in_queue: asyncio.Queue, out_queue: asyncio.Queue):
            while True:
                item = await in_queue.get()
                if item is None:
                    in_queue.task_done()  # Важно! Помечаем задачу как выполненную для сигнала завершения
                    break
                
                try:
                    result = await self._process_item(item)
                    await out_queue.put(result)
                except Exception as e:
                    logger.error(f"Ошибка при обработке записи с __index__ {item.get('__index__', 'N/A')}: {e}")
                    await out_queue.put(e) # Помещаем ошибку в очередь для подсчета
                finally:
                    in_queue.task_done()

        input_queue = asyncio.Queue(maxsize=self.max_concurrent_tasks)
        output_queue = asyncio.Queue()
        
        # Запускаем worker-ы
        workers = [asyncio.create_task(worker(input_queue, output_queue)) for _ in range(self.max_concurrent_tasks)]

        processed_count = 0
        # Порядок выдачи результатов
        expected_index = ds[0]['__index__'] if len(ds) > 0 and '__index__' in ds.column_names else 0
        pending: Dict[int, Dict[str, Any]] = {}

        async def producer():
            for item in ds:
                await input_queue.put(item)
            
            # Отправляем сигналы о завершении для worker-ов
            for _ in range(self.max_concurrent_tasks):
                await input_queue.put(None)
        
        producer_task = asyncio.create_task(producer())

        while processed_count < total_items:
            result = await output_queue.get()
            processed_count += 1
            pbar.update(1)
            
            if not isinstance(result, Exception):
                idx = result.get('__index__')
                if idx is None:
                    yield result
                else:
                    pending[idx] = result
                    while expected_index in pending:
                        yield pending.pop(expected_index)
                        expected_index += 1
        
        await producer_task
        await input_queue.join()
        await asyncio.gather(*workers)
        
        pbar.close()
