# krist-synth

<img src="assets/logo.png" alt="logo" width="200">

Гибкий фреймворк для генерации синтетических данных. Не особо преднозначено для использования. Возможно когда-то будет готово к настоящему использованию.

## Как устанавливать

[Надо установить uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
git clone https://github.com/kristaller486/krist-synth
cd krist-synth
uv sync
```

## Как запускать

```bash
uv run python -m src.pipeline.runner --pipeline-config <config path>
```