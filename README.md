# VEGAS Subtitle Automation

Локальный прототип для субтитров игровых роликов в VEGAS Pro. Извлекает готовые
титры и исходные аудиоинтервалы из `.veg`, собирает кандидатов для датасета,
сравнивает ASR-модели на отложенных записях и применяет проверенный план в VEGAS.

## Что работает

- Чтение проектов через штатный `ScriptPortal.Vegas`, без изменения исходников.
- Восстановление ссылок на исходный звук, включая участки, заглушённые в монтаже.
- Отсев дублей, пересечений голосов, маскированных и нерусских подписей, изменённой скорости.
- Разделение train/validation/test по связанным исходным записям.
- GPU-бенчмарк Whisper через faster-whisper и подсчёт WER.
- Парное сравнение Whisper, GigaAM и Qwen3-ASR на общем WAV-наборе, WER и CER.
- Дополнительные GPU-бэкенды: NVIDIA Parakeet TDT, T-one ONNX и русская адаптация Whisper.
- Раздельные оценки validation и development, дополнительные словарные диагностики мата.
- Частная очередь проверки с волной, выделением и автосохранением.
- Предложения временных границ от Qwen ForcedAligner и подсказки двух ASR.
- Экспорт только явно проверенных аудио-текстовых пар для обучения.
- Создание редактируемых титров, цензуры и хотбара по проверенному плану.
- Плавные края всех субтитров и измерение боковых отступов по рендеру.
- Docker Compose для датасета и GPU-распознавания.

Автоматическое определение говорящего, окончательный поиск мата и обучение
адаптера ещё не объединены в готовый конвейер. Это основа для измеримого
улучшения, а не обещание безошибочной расшифровки одним запуском.

## Архитектура

VEGAS и C#-скрипты работают в Windows. Linux-контейнеры получают XML-экспорт и
исходные записи через тома только для чтения. Модели, датасет и результаты
хранятся в `local/`, которая исключена из Git и контекста Docker-сборки.
Медиа не отправляются во внешние API. Загрузка моделей обращается к их хостингу.

## Docker Compose

Нужны Docker Desktop с Linux-контейнерами и доступом к NVIDIA GPU.
Тома D и F задаются в `.env` по примеру `.env.example`. В Linux/WSL используйте
пути `/mnt/d` и `/mnt/f`. Диски подключаются только для чтения.

```powershell
docker compose --profile gpu build
docker compose run --rm tools manifest
docker compose run --rm tools download-model large-v3
docker compose --profile gpu run --rm --entrypoint nvidia-smi asr
docker compose --profile gpu run --rm asr benchmark `
  --model local/models/large-v3 --limit 30 `
  --output local/benchmarks/large-v3
docker compose run --rm --entrypoint python tools -m unittest discover -s tests -v
```

Контейнеры одноразовые. `docker compose up` для постоянной службы не требуется.
Не выполняйте `docker system prune` или удаление томов ради этого проекта.
Модели загружаются по commit revision и SHA256 из `configs/models.lock.json`.
Повторная команда проверяет имеющиеся файлы, без повторного скачивания.
Замеры и выбор следующих кандидатов описаны в `docs/MODELS.md`.

## Сравнение и разметка

В пилотном сравнении выбран GigaAM v3 RNN-T. Подробности и ограничения:
`docs/MODELS.md`. `research` содержит отдельное PyTorch CUDA окружение,
основной контейнер CTranslate2 остаётся небольшим и независимым от него.

После ручной разметки сравнивались 12 конфигураций 11 моделей на 98 общих
фрагментах. Из них только 14 относятся к validation. Остальные предназначены
для будущего обучения. Все 98 вместе служат диагностикой, независимым тестом
такой результат не является. Holdout остаётся исключённым.

```powershell
docker compose --profile research build research
docker compose run --rm tools download-model gigaam-rnnt
docker compose run --rm tools download-model qwen3-asr-1.7b
docker compose run --rm tools prepare-evaluation
docker compose --profile research run --rm research evaluate `
  --engine gigaam --model local/models/gigaam-rnnt `
  --output local/comparison/gigaam-rnnt
docker compose run --rm tools stage-dataset
docker compose run --rm tools prepare-review-batch --limit 120
docker compose --profile review up -d --wait review
```

Страница проверки: `http://127.0.0.1:8765`. Команда подготовки очереди создаёт
`local/review-batches/current.json` с ID из train/validation и доступными подсказками.
Существующая очередь не перезаписывается. Для следующей укажите новый `--output`.
Сохранение: `local/reviews/batch-001.json`. База аудио и хеши не меняются во время
ручной проверки. Повторный `stage-dataset` требует нового выходного каталога.

Текст можно проверить отдельно от границ. При перемещении границ оценивается
итоговое выделение. Если все слова слышны целиком, выбирается полное покрытие
и подтверждаются границы. Обрывок слова нельзя дописывать по смыслу.

```powershell
docker compose run --rm tools export-reviewed `
  --review local/reviews/batch-001.json --output local/training-reviewed-v1
```

JSONL сохраняет исходный проверенный текст. TSV предназначен для GigaAM и
содержит только строки с поддерживаемым кириллическим алфавитом. Цифры и
латинские названия требуют проверки произнесённой формы. Список исключений
сохраняется отдельно. См. `docs/TRAINING.md`.

Экспорт также создаёт `development.jsonl` из train/validation. Для диагностики
его можно подготовить с `prepare-reviewed-evaluation --split development`.
Параметр указан явно, чтобы диагностический набор не подменял validation.
Итоговый отчёт собирается командой:

```powershell
docker compose run --rm tools comparison-report `
  --runs local/comparison-expanded --output local/reports/expanded
```

Команда проверяет общий набор и создаёт JSON/HTML. Словарный подсчёт семейств
мата не измеряет качество запикивания или ошибки временных границ.

## Экспорт проектов в Windows

Установите Python 3.12 и пакет для управляющих команд:

```powershell
python -m venv .venv-win
.venv-win\Scripts\python -m pip install -e .
.venv-win\Scripts\vegas-subtitles export-corpus `
  --vegas-exe 'C:\Program Files\VEGAS\Vegas Pro 23\vegas230.exe' `
  --projects-dir 'D:\vegas' `
  --exclude 'unfinished-project' --exclude 'generated-project'
```

Команда открывает один отдельный экземпляр VEGAS. Он последовательно читает
проекты в фоне, записывает `local/corpus/projects/*.xml` и закрывается.
Ход работы: `local/corpus/progress.log`, завершение: `complete.txt`.
При повторном запуске уже экспортированные файлы пропускаются. Для обновления
конкретного экспорта перенесите его XML в другую папку и повторите экспорт.

## Применение проверенного плана

Формат показан в `configs/review.example.json`. Данные содержат точные интервалы,
говорящего и уже отредактированный текст. Мат в тексте должен быть маскирован,
а в `bleeps` указаны интервалы для заглушения оригинала и сигнала.

```powershell
.venv-win\Scripts\vegas-subtitles prepare-job `
  --source 'D:\vegas\unfinished.veg' `
  --review 'local\review.json' --output 'D:\vegas\finished.veg' `
  --reference 'D:\vegas\reference.veg' --bleep 'F:\sounds\beep.mp3'
.venv-win\Scripts\vegas-subtitles run-job `
  --vegas-exe 'C:\Program Files\VEGAS\Vegas Pro 23\vegas230.exe'
```

Текущий адаптер ожидает структуру: две пустые видеодорожки для титров сверху,
основное видео на третьей дорожке, диалог на четвёртой и пустая аудиодорожка
для сигнала на пятой. При неверной структуре сборка останавливается. Из
reference берётся хотбар с видеодорожки с индексом 2. Индексы считаются с нуля.
Поддерживаемый профиль: 1440×2560, 60000/1001 fps. Универсальный поиск дорожек
и других геометрий пока не реализован.

Исходный проект копируется в `local/jobs/current/input.veg`. Редактор изменяет
только эту временную копию и сохраняет результат в новый выходной путь.
Путь вывода не должен уже существовать. Завершение проверяйте по
`build-result.txt`, ошибки по `build-error.txt`. Исходные медиа не изменяются.

## Данные и качество

`candidate=true` означает пригодность для дальнейшей проверки, а не разрешение
на обучение. У всех строк первоначально `training_approved=false`.
Границы монтажных титров нужно проверить или выровнять по речи. Подписи с
`бл**ь` нельзя использовать как буквальную расшифровку произнесённого слова.
Сначала восстанавливается правильное слово. См. `docs/TRAINING.md`.

WER измеряет различие слов после нормализации регистра, пунктуации и `ё/е`.
Он не измеряет правильность говорящего, запикивания, оформления или смысла.
На коротких монтажных подписях ошибки нарезки также повышают WER.

## Разработка

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v
```

`configs/profile.example.json` описывает целевой профиль. В C#-адаптере v0.1
этот профиль ещё зафиксирован константами, JSON служит спецификацией.
До публичной публикации выберите лицензию и проверьте состав `git ls-files`.
