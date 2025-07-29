# src/pipeline/batcher/static_batcher.py
import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, List

from datasets import Dataset
from tqdm.auto import tqdm

from src.pipeline.batcher.base_batcher import BaseBatcher

logger = logging.getLogger(__name__)


class StaticBatcher(BaseBatcher):
    def __init__(self, steps: List[Any], config: Dict[str, Any]):
        super().__init__(steps, config)
        self.batch_size = self.config.get('parallel_batch_size', 10)

    async def process(self, ds: Dataset) -> AsyncGenerator[Dict[str, Any], None]:
        for i in tqdm(range(0, len(ds), self.batch_size), desc="Обработка батчей"):
            batch_items = [ds[j] for j in range(i, min(i + self.batch_size, len(ds)))]
            tasks = [self._process_item(item) for item in batch_items]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            successful_count = 0
            for item, result in zip(batch_items, batch_results):
                if isinstance(result, dict):
                    successful_count += 1
                    yield result
                else:
                    logger.error(f"Ошибка при обработке записи с __index__ {item['__index__']}: {result}")
            
            logger.info(f"Батч завершен. Успешно: {successful_count}/{len(batch_items)}")
