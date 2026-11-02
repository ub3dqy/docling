# Разбор ошибок русского распознавания

Это разбор уже проверенного комплекта C, а не новая независимая выборка.
Исходные документы, ответы, области проверки и оценки первого прогона сохранены.
Обучение, замена весов, установка зависимостей и изменения основной программы
не выполнялись.

См. [отчёт](REPORT.md). Инструменты для повторения:

- `scripts/diagnose_russian_ocr.py` сохраняет результаты между этапами обработки.
- `scripts/probe_russian_ocr_regions.py` сравнивает существующие распознаватели
  на заранее размеченных участках либо на сохранённых автоматически найденных
  строках. Это отдельные опыты без сборки готового документа.

Все опыты запускаются последовательно, на CPU с четырьмя потоками,
с локальными моделями `model_artifacts/ru` и заблокированными соединениями Python.
Папка результата должна быть новой. Подробные промежуточные данные и изображения
остаются в локальной папке `output/russian-ocr-diagnosis-2026-11-02`.

## Сохранённые данные

- `stage-results.json`, `field-stages.json`: исходные чтения и прохождение
  тех же 60 числовых полей через этапы.
- `controlled-comparisons.json`: два отдельных опыта с определением структуры.
- `lines-*.json`: сравнение трёх распознавателей на 309 найденных строках.
- `regions-*.json`: первоначальный опыт с широкими контрольными областями;
  его неудовлетворительные результаты также сохранены.
- `verification.json`: сохранность программы, весов, ответов и десяти выгрузок C.
- `artifact-hashes.json`: контрольные суммы подробных местных наблюдений.

## Повторение на этом проекте

Исходники комплекта C должны быть загружены и совпадать с его `sources.json`.
Команды выполняются из корня проекта. Ни одна из них не меняет профиль `ru`.
Примеры создают новые папки; для следующего повторения укажите другое имя.

```powershell
$refPath = 'tests/data/russian_ocr_holdout_20261102c/references.json'
$manifestPath = 'tests/data/russian_ocr_holdout_20261102c/sources.json'
$sourcePath = 'output/russian-ocr-holdout-2026-11-02c/sources'

uv run --no-sync python -m scripts.diagnose_russian_ocr --references $refPath --manifest $manifestPath --sources $sourcePath --artifacts-path model_artifacts/ru --output output/diagnosis-repeat/full

uv run --no-sync python -m scripts.diagnose_russian_ocr --references $refPath --manifest $manifestPath --sources $sourcePath --artifacts-path model_artifacts/ru --output output/diagnosis-repeat/tableformer --variant tableformer --case mosreg-assets

uv run --no-sync python -m scripts.diagnose_russian_ocr --references $refPath --manifest $manifestPath --sources $sourcePath --artifacts-path model_artifacts/ru --output output/diagnosis-repeat/linear-layout --variant linear-layout --case receipt-canteen-507

uv run --no-sync python -m scripts.probe_russian_ocr_regions --references $refPath --manifest $manifestPath --sources $sourcePath --artifacts-path model_artifacts/ru --trace-run output/diagnosis-repeat/full --output output/diagnosis-repeat/detected-lines

uv run --no-sync python -m scripts.probe_russian_ocr_regions --references $refPath --manifest $manifestPath --sources $sourcePath --artifacts-path model_artifacts/ru --output output/diagnosis-repeat/control-regions
```

Сценарии — инструменты последовательной диагностики в отдельном процессе.
Они временно наблюдают внутренние этапы программы; не подключайте их
к обслуживающему параллельные запросы приложению.
