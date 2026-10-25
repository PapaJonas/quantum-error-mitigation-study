# Сирі датасети (Василь Сольчаник)

Два додаткові синтетичні датасети, незалежні від `../synthetic_expectations.csv`
(той набір описаний в `../README.md` і використовується в `notebooks/eda_baseline_models.ipynb`).
Код генерації: [`src/data_generation/`](../../src/data_generation/).

Файли збережені через [Git LFS](https://git-lfs.com/) — `.csv` у цій папці не лежать у звичайній
git-історії як текст, потрібен встановлений `git-lfs` (`git lfs install`) перед клонуванням/пулом.

## `single_qubit_dataset.csv`

315 000 рядків: 5000 унікальних одно-кубітних схем (1–10 випадкових гейтів з `RX`/`RY`/`RZ`/`H`,
кути частково фіксовані з `BASE_ANGLES`, частково випадкові) × 21 рівень depolarizing-шуму
(0.001–0.050) × 3 варіанти `shots` (256 / 1024 / 4096). Ціль регресії — `delta = ideal_expval -
noisy_expval`, однокубітний спостережуваний `Z`. Код: `src/data_generation/dataset_builder.py`,
`generate_dataset(n_circuits=5000, seed=42)`.

## `multiqubit_dataset.csv`

185 400 рядків: унікальні схеми на 2–8 кубітів (кількість схем на розмір — від 60 для 2 кубітів до
10 для 8), з 8 "сімейств" (`ghz`, `bell_pairs`, `rotation_cx_ladder`, `rotation_cx_ring`,
`random_rotation_cx`, `qft_like`, `bv_like`, `cx_dense_random`), 1–3 шари гейтів. Для кожної схеми
рахуються всі одно-кубітні `Z_i` та сусідні `ZZ_i,i+1` спостережувані (тому на одну схему припадає
кілька рядків), під 5 рівнів depolarizing-шуму (0.001–0.02) × 3 варіанти `shots`. Код:
`src/data_generation/multiqubit_dataset_generator.py`, `generate_dataset(seed=42)`.

## Спільна схема колонок

`record_id, circuit_id, circuit_type, n_qubits, circuit_depth, single_qubit_gate_count, cx_count,
mean_rotation_angle, observable, circuit_params, noise_category, noise_params, backend_name, shots,
seed, ideal_expval, noisy_expval, delta, timestamp`. `circuit_params` і `noise_params` — JSON-рядки
з повним описом гейтів/шуму. Перевірено: жодних дублікатів `record_id`, кількість рядків відповідає
очікуваній (кожен генератор сам звіряє це при запуску й падає з помилкою при розбіжності).

## Відоме обмеження — поки не зроблено

На відміну від `../synthetic_expectations.csv`, тут **немає зафіксованого поділу train/validation/test**
і немає `manifest.json` з SHA-256/версіями бібліотек. Обидва генератори детерміновані (`seed=42`
за замовчуванням, усі випадкові кроки йдуть через один `random.Random(seed)`), тому відтворити дані
можна точно — але сам поділ на вибірки для ML ще треба спроєктувати й зафіксувати перед тим, як
тренувати на цих даних модель (важливо: `circuit_id` тут інший формат, ніж у головному датасеті, і
при розбитті на train/test його так само не можна розділяти між вибірками).

## Відтворення

```
uv run python -c "from src.data_generation.dataset_builder import generate_dataset; generate_dataset(n_circuits=5000, seed=42)"
uv run python -c "from src.data_generation.multiqubit_dataset_generator import generate_dataset; generate_dataset(seed=42)"
```
