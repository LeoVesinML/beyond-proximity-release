# SemanticSplat — команды терминала

Рабочая папка: корень `beyond-proximity`.

---

## 1. Подготовка (один раз за сессию)

```powershell
cd "C:\Users\Asus\Desktop\Semantic 3D mapping\beyond-proximity"
```

```powershell
.\.venv\Scripts\Activate.ps1
```

```powershell
$env:PYTHONPATH = "."
```

---

## 2. Проверить данные Nikita (v2)

```powershell
(Get-ChildItem data\replica\pilot_scene_001\rgb).Count
```

```powershell
(Get-ChildItem data\replica\pilot_scene_001\depth).Count
```

```powershell
Test-Path data\replica\pilot_scene_001\poses.json
```

Ожидается: `19`, `19`, `True`.

---

## 3. Импорт сцены в backend

```powershell
python scripts\import_replica_scene.py `
  --input data\replica\pilot_scene_001 `
  --scene-id replica_pilot_001 `
  --copy-annotations-from default `
  --backend-root .
```

Проверка:

```powershell
Test-Path backend\data\scenes\replica_pilot_001\transforms.json
```

```powershell
Get-Content backend\data\scenes\replica_pilot_001\transforms.json | Select-Object -First 15
```

---

## 4. Smoke test pipeline (stub)

Полный прогон (с импортом):

```powershell
python scripts\run_pipeline.py `
  --scene data\replica\pilot_scene_001 `
  --scene-id replica_pilot_001 `
  --query "find the stage" `
  --out outputs\week1_smoke_test_v2 `
  --mode stub `
  --copy-annotations-from default
```

Без переимпорта:

```powershell
python scripts\run_pipeline.py `
  --scene data\replica\pilot_scene_001 `
  --scene-id replica_pilot_001 `
  --query "find the chair near the table" `
  --out outputs\week1_smoke_test_v2_chair `
  --mode stub `
  --skip-import
```

Проверка outputs:

```powershell
Get-ChildItem outputs\week1_smoke_test_v2
```

```powershell
Get-Content outputs\week1_smoke_test_v2\query_result.json -Encoding utf8
```

---

## 5. Пересобрать pilot scene из demo default (опционально)

```powershell
python scripts\prepare_replica_scene.py `
  --input backend\data\scenes\default `
  --output data\replica\pilot_scene_001
```

С ресайзом под 800×600:

```powershell
python scripts\prepare_replica_scene.py `
  --input backend\data\scenes\default `
  --output data\replica\pilot_scene_001 `
  --image-policy resize_to_intrinsics
```

---

## 6. Backend-тесты (опционально)

```powershell
pytest tests\backend\ -v
```

---

## 7. Запуск demo UI (не нужен для pipeline)

**Вариант A — одной командой (3 окна):**

```powershell
cd "C:\Users\Asus\Desktop\Semantic 3D mapping\beyond-proximity"
.\start_all.cmd
```

**Вариант B — вручную в трёх терминалах:**

```powershell
cd "C:\Users\Asus\Desktop\Semantic 3D mapping\beyond-proximity"
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "."
python -m backend.api.server
```

**Терминал 2 — MCP:**

```powershell
cd "C:\Users\Asus\Desktop\Semantic 3D mapping\beyond-proximity"
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "."
python -m backend.mcp.server
```

**Терминал 3 — Frontend:**

```powershell
cd "C:\Users\Asus\Desktop\Semantic 3D mapping\beyond-proximity\frontend"
npm run dev
```

Браузер: http://localhost:5173

---

## 8. Benchmark Leo (graph vs flat) — Windows

После merge с fork Leo. Один раз установить matplotlib (уже в `backend/requirements.txt`):

```powershell
cd "C:\Users\Asus\Desktop\Semantic 3D mapping\beyond-proximity"
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "."
pip install -r backend\requirements.txt
```

Полный прогон (отчёты, графики, reasoning):

```powershell
.\run_all_docs.ps1
```

Или через cmd:

```powershell
cmd /c run_all_docs.cmd
```

Или вручную:

```powershell
$env:MPLCONFIGDIR = ".\.matplotlib"
python scripts\run_full_benchmark.py
```

Стартовая страница доков: `docs\README.md`

**Примечание:** генерация MP4 требует `ffmpeg` в PATH. Без ffmpeg видео пропускается, остальное работает (в репо уже есть готовый `docs/benchmark_results/failure_case_demo.mp4` от Leo).

---

## 9. Git

```powershell
cd "C:\Users\Asus\Desktop\Semantic 3D mapping\beyond-proximity"
git status
```

```powershell
git push -u origin main
```

---

## Примечания

- Бэкап до merge: `..\local_backup_before_leo_merge_2026-06-09.zip` (вне git, см. `..\`.gitignore`).
- Клон Leo для сравнения: `..\beyond-proximity-leo\` (тоже вне git).
- Большие `rgb/` и `depth/` в git не попадают (см. `.gitignore`). Локально должны лежать в `data/replica/pilot_scene_001/`.
- Stub использует готовые `views/` и `tree/` из `default` — для live mode нужен отдельный этап.
- Depth в pilot scene сейчас константа `0.1` — 3D bbox в stub не работает.
