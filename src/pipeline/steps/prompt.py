# src/pipeline/steps/prompt.py
from jinja2 import Environment, FileSystemLoader
from .base_step import BaseStep
from pathlib import Path

class FormatPromptStep(BaseStep):
    """
    Шаг для форматирования промпта с использованием шаблона Jinja2 из файла.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.template_path = self.config['template_path']
        self.output_key = self.config['output_key']
        
        # Инициализируем Jinja Environment
        # Предполагаем, что template_path - это полный путь к файлу.
        # Загрузчик будет работать с директорией, где лежит шаблон.
        path_obj = Path(self.template_path)
        template_dir = path_obj.parent
        template_name = path_obj.name
        self.jinja_env = Environment(loader=FileSystemLoader(searchpath=template_dir), autoescape=False)
        self.template = self.jinja_env.get_template(template_name)

    async def process(self, item: dict) -> dict:
        """
        Рендерит шаблон, используя данные из item, и добавляет результат в item.
        """
        try:
            # item сам используется как контекст для рендеринга
            formatted_prompt = self.template.render(item)
            item[self.output_key] = formatted_prompt
        except Exception as e:
            self.logger.error(f"Ошибка при рендеринге шаблона {self.template_path}: {e}")
            # В случае ошибки добавляем None, чтобы пайплайн мог это обработать
            item[self.output_key] = None
        
        return item
