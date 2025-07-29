# src/pipeline/batcher/base_batcher.py
import asyncio
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Coroutine, List, Dict

from datasets import Dataset
from tqdm.auto import tqdm

from src.pipeline.steps.base_step import BaseStep


class BaseBatcher(ABC):
    def __init__(self, steps: List[BaseStep], config: Dict[str, Any]):
        self.steps = steps
        self.config = config

    @abstractmethod
    async def process(self, ds: Dataset) -> AsyncGenerator[Dict[str, Any], None]:
        ...

    async def _process_item(self, item: dict) -> dict:
        for step in self.steps:
            item = await step.process(item)
        return item
