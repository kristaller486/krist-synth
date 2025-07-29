# src/pipeline/runner.py
import argparse
import logging
import sys
import asyncio
import json
from pathlib import Path
import importlib
from datasets import load_dataset, Dataset
import yaml
import os
import time
import shutil

from src.pipeline.steps.base_step import BaseStep
from src.pipeline.batcher.base_batcher import BaseBatcher

# --- Настройка логирования ---
# Устанавливаем базовый уровень INFO
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)

# Включаем DEBUG для конкретных модулей
logging.getLogger("src.pipeline.runner").setLevel(logging.DEBUG)
logging.getLogger("src.pipeline.steps.translation").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def dynamic_import_class(class_path: str) -> type:
    try:
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError, ValueError) as e:
        logger.error(f"Не удалось импортировать класс по пути: {class_path}")
        raise ImportError(f"Класс {class_path} не найден.") from e

def get_manifest_path(checkpoint_dir: Path, pipeline_name: str) -> Path:
    return checkpoint_dir / f".{pipeline_name}.manifest.json"

async def main():
    parser = argparse.ArgumentParser(description="Запуск гранулярного пайплайна обработки данных.")
    parser.add_argument('--pipeline-config', type=str, required=True, help='Путь к YAML файлу конфигурации пайплайна.')
    args = parser.parse_args()

    try:
        config = load_config(args.pipeline_config)
        pipeline_name = config.get("pipeline_name", "unnamed_pipeline")
        
        # --- 1. Загрузка данных ---
        data_config = config.get('data_source', {})
        ds = load_dataset(data_config['path'], split=data_config.get('split', 'train'))
        ds = ds.add_column("__index__", range(len(ds)))
        logger.info(f"Загружено {len(ds)} записей, добавлен столбец '__index__'.")
        
        if 'take_first_n' in data_config and data_config['take_first_n'] > 0:
            n = data_config['take_first_n']
            ds = ds.select(range(min(n, len(ds))))
            logger.info(f"Взят срез из первых {len(ds)} записей.")

        # --- 2. Настройка возобновления через чекпоинты ---
        checkpoints_config = config.get('checkpoints', {})
        if not checkpoints_config or 'path' not in checkpoints_config:
            raise ValueError("Конфигурация чекпоинтов ('checkpoints.path') не определена.")

        checkpoint_dir = Path(checkpoints_config['path'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file_path = checkpoint_dir / f"{pipeline_name}.jsonl"
        manifest_path = get_manifest_path(checkpoint_dir, pipeline_name)
        last_processed_index = -1

        if checkpoint_file_path.exists() and manifest_path.exists():
            try:
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)
                    last_processed_index = manifest.get('last_processed_index', -1)
                if last_processed_index > -1:
                    logger.info(f"Обнаружен манифест. Возобновление работы с индекса {last_processed_index + 1}.")
                    ds = ds.filter(lambda item: item['__index__'] > last_processed_index, num_proc=os.cpu_count())
                    logger.info(f"Осталось обработать {len(ds)} записей.")
            except (IOError, json.JSONDecodeError):
                logger.warning("Манифест поврежден или нечитаем. Начинаем обработку с нуля.")
                checkpoint_file_path.unlink(missing_ok=True) # Удаляем поврежденный чекпоинт
        
        # --- 3. Инициализация шагов ---
        pipeline_steps = []
        for step_conf in config.get('steps', []):
            StepClass = dynamic_import_class(step_conf['class_path'])
            pipeline_steps.append(StepClass(config=step_conf.get('config', {})))

        # --- 4. Выполнение пайплайна ---
        processing_config = config.get('processing', {})
        batcher_class_path = processing_config.get('batcher_class', 'src.pipeline.batcher.static_batcher.StaticBatcher')
        BatcherClass = dynamic_import_class(batcher_class_path)
        
        if not issubclass(BatcherClass, BaseBatcher):
            raise TypeError(f"Класс батчера '{batcher_class_path}' должен наследоваться от BaseBatcher.")
            
        batcher = BatcherClass(steps=pipeline_steps, config=processing_config)
        
        with open(checkpoint_file_path, 'a', encoding='utf-8') as f_checkpoint:
            async for result in batcher.process(ds):
                if isinstance(result, dict) and '__index__' in result:
                    f_checkpoint.write(json.dumps(result, ensure_ascii=False) + '\n')
                    
                    # Обновление манифеста после каждой успешной записи
                    last_processed_index = result['__index__']
                    with open(manifest_path, 'w', encoding='utf-8') as f_manifest:
                        json.dump({'last_processed_index': last_processed_index}, f_manifest)
                else:
                    logger.warning(f"Получен некорректный результат от батчера: {result}")
        
        # --- 5. Финальное сохранение ---
        logger.info("Обработка завершена. Выполняется финальное сохранение...")
        output_config = config.get('output', {})
        output_format = output_config.get('format', 'jsonl')
        output_path_str = output_config.get('path')

        if not output_path_str:
            logger.warning("Секция 'output' или 'output.path' не определена. Пропускаем финальное сохранение.")
        else:
            if not checkpoint_file_path.exists() or checkpoint_file_path.stat().st_size == 0:
                logger.warning(f"Файл с результатами {checkpoint_file_path} пуст или не существует. Пропускаем финальное сохранение.")
            else:
                logger.info(f"Загрузка результатов из {checkpoint_file_path} для финального сохранения.")
                final_ds = load_dataset('json', data_files=str(checkpoint_file_path), split='train')
                
                if '__index__' in final_ds.column_names:
                    final_ds = final_ds.remove_columns(['__index__'])

                logger.info(f"Сохранение {len(final_ds)} записей в формате '{output_format}' по пути: {output_path_str}")

                if output_format == 'jsonl':
                    final_output_path = Path(output_path_str)
                    final_output_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(checkpoint_file_path, final_output_path)
                    logger.info(f"Результаты сохранены в {final_output_path}")

                elif output_format == 'dataset':
                    try:
                        if output_config.get('push_to_hub', False):
                            logger.info(f"Публикация датасета на Hugging Face Hub: {output_path_str}")
                            final_ds.push_to_hub(
                                repo_id=output_path_str,
                                private=output_config.get('private', False)
                            )
                            logger.info("Датасет успешно опубликован.")
                        else:
                            logger.info(f"Сохранение датасета на диск: {output_path_str}")
                            final_ds.save_to_disk(output_path_str)
                            logger.info("Датасет успешно сохранен на диск.")
                    except Exception as e:
                        logger.error(f"Ошибка при сохранении датасета: {e}", exc_info=True)
                        logger.error("Убедитесь, что вы вошли в huggingface-cli (`huggingface-cli login`) и имеете права на запись в репозиторий.")
                else:
                    logger.error(f"Неизвестный формат вывода: {output_format}. Допустимые значения: 'jsonl', 'dataset'.")
        
        logger.info("Пайплайн успешно завершен.")

    except Exception as e:
        logger.exception(f"Произошла непредвиденная ошибка: {e}")

if __name__ == "__main__":
    import os
    asyncio.run(main())
