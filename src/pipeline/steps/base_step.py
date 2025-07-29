# src/pipeline/steps/base_step.py
from abc import ABC, abstractmethod
from datasets import Dataset
import logging

class BaseStep(ABC):
    """
    Абстрактный базовый класс для шага в пайплайне обработки данных.
    Каждый шаг выполняет операцию над одним элементом данных (словарем).
    """

    def __init__(self, config: dict):
        """
        Инициализирует шаг.

        :param config: Конфигурация для этого конкретного шага.
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def process(self, item: dict, batch_id: int = 0) -> dict:
        """
        Выполняет основную логику шага над одним элементом данных.

        :param item: Словарь, представляющий одну строку данных.
        :param batch_id: Идентификатор текущего батча для логирования.
        :return: Модифицированный словарь.
        """
        pass
