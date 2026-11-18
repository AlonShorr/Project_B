# t5 — כל הפרמטרים במקום אחד, והרצה בלי GUI

> הסבר לקוד של `lidar_sim/configs/default.yaml`, `sim_config.py`, `policies.py`, `runner.py`, `test_runner.py`.
> כל הדוגמאות כאן הורצו בפועל על הענף `motion-model-t3`.

## 1. איזו בעיה זה פותר

כל התוצאות בפרויקט הן השוואות: אלגוריתם A מול B, רעש נמוך מול רעש גבוה, `τ_declare` אחד מול אחר (עקומת Pareto ב-t8).
השוואה כזאת שווה משהו רק אם (א) יודעים **בדיוק** עם אילו פרמטרים ועל איזה מבוך כל תוצאה הורצה, (ב) הרצה חוזרת נותנת **בדיוק** את אותו מספר,
ו-(ג) אפשר להריץ אלפי אפיזודות מהר, בלי חלון pygame. t5 נותן את שלושת הדברים: קובץ YAML אחד עם כל הפרמטרים,
seed לכל אפיזודה, ו-`runner.py` שמריץ אפיזודות "headless" (בלי ממשק גרפי) וכותב שורה אחת ל-CSV לכל אפיזודה.

## 2. התמונה הגדולה

```
configs/default.yaml   ─┐
קובץ ניסוי (--config)   ─┼─► load_config() ─► SimConfig  (קפוא + מאומת)
--set / --n-runs / ...  ─┘                       │
                                                  ├─► resolve_maze() ─► LoadedMaze  (מתוך maze_bank)
                                                  ▼
                              EpisodeRunner = GridEnv + Policy   (נבנה פעם אחת לכל worker)
                                                  │
                         לכל seed בטווח  seed0 .. seed0+n_runs-1:
                                                  ▼
            env.reset(seed) → policy.reset(env_info, policy_rng(seed))
            while not env.done:  action = policy.act(obs, info);  env.step(action)
                                                  ▼
                         שורה אחת ב-CSV   +   קובץ sidecar  <out>.config.yaml
```

- `sim_config.py` — קורא YAML והופך אותו לאובייקט `SimConfig` (dataclasses קפואים, כלומר אי אפשר לשנות אותם בטעות באמצע ריצה).
  ל-dataclasses **אין ערכי ברירת מחדל**: `default.yaml` הוא המקור היחיד שלהם.
- `grid_env.py` — הסביבה (מ-t3): מגרילה תא התחלה, מזיזה את הרובוט עם הרעש, ומחליטה מתי האפיזודה נגמרת.
- `policies.py` — "המוח": מקבל מה שהרובוט יודע ומחזיר פעולה.
- `runner.py` — מחבר הכול, מריץ, כותב CSV.

אפיזודה נגמרת באחת מארבע תוצאות (`Outcome`): `success` (הכריז `DECLARE` בתא מטרה), `wrong_declaration` (הכריז במקום אחר),
`collision` (נכנס בקיר כש-`wall_mode=collide`), `timeout` (הגיע ל-`max_steps` פעולות).

## 3. קובץ ההגדרות

### הסעיפים ומה כל פרמטר אומר על הרובוט

| סעיף | פרמטר | משמעות |
|---|---|---|
| `maze` | `name` | איזה מבוך לטעון מ-`maze_bank` (למשל `medium/m10_a`, `small/m5_a`) |
| | `goal` | `null` = תאי המטרה של המבוך עצמו, או רשימה `[[r, c], ...]` |
| `motion` | `p_steps` | ההסתברות שפקודת תזוזה תזיז את הרובוט `k = 0, 1, 2` תאים (0 = החליק ונשאר, 2 = "עף" תא נוסף). אם `null` — מחושב מחמשת הפרמטרים הבאים; אם נתון — הם מתעלמים |
| | `cell_size_mm` | גודל תא (180 מ"מ): אורך הצעד המתוכנן, ורוחב ה"סל" (±חצי תא) של כל `k` |
| | `step_sigma_mm`, `step_bias_mm` | סטיית תקן והטיה (מ"מ) של אורך הצעד, בצורת הגאוסיאן של פרויקט A |
| | `extra_stay`, `extra_over` | מסה נוספת ל-`k=0` (החלקה/היתקעות, `0.05`) ול-`k=2` (גלישה, `0.0001`). הסכום חייב להיות ≤ 0.95. ביחד עם הנ"ל יוצא בערך `(0.05, 0.9499, 0.0001)` |
| | `p_drift` | הסתברות לסטות 45° מהכיוון שנצטווה (חצי לכל צד). `0.0001` = שגיאה גסה ונדירה |
| | `diag_rule` | **תיעוד בלבד** של החלטה D2: צעד אלכסוני חוקי רק אם כל 4 הדפנות שנפגשות בפינה פתוחות. הכלל מקודד ב-`can_step`, ואין ערך אחר מותר |
| | `wall_mode` | `collide` = פגיעה בקיר מסיימת את האפיזודה בכישלון; `stop` = הרובוט עוצר מול הקיר וממשיך (ההתנהגות של פרויקט A) |
| `sensing` | `R`, `rays`, `p_miss`, `miss_mode` | טווח הקרן בתאים, 4 או 8 קרניים, הסתברות "לפספס" קיר, ומה רואים אחרי פספוס. **עוד לא בשימוש** (t4) |
| `episode` | `max_steps` | מקסימום פעולות לפני `timeout` (`400`) |
| | `step_cost`, `declare_cost` | העלות `T`: `step_cost` לכל תזוזה, `declare_cost` להכרזה |
| `projecta` | `tau_loc`, `rho_amb`, `tau_lost`, `sep_min`, `tau_declare`, `eps_col`, `move_cost` | הספים של היוריסטיקת פרויקט A: מתי "ממוקם" (`tau_loc`, ומתחרה במרחק ≥ `sep_min` תאים קטן מ-`rho_amb` ממנו), מתי "אבוד" (`tau_lost`, חייב להיות קטן מ-`tau_loc`), מתי להכריז (`tau_declare`), כמה סיכון התנגשות מותר לצעד (`eps_col = 0.005`). **עוד לא בשימוש** (t7) |
| `policy` | `name`, `params` | איזו מדיניות להריץ (`random` / `oracle_bfs`) ופרמטרים לכל אחת. רק הכניסה של `name` נכנסת לשימוש |
| `run` | `n_runs`, `seed0`, `workers`, `out` | כמה אפיזודות, seed ראשון, כמה תהליכים במקביל, לאן לכתוב את ה-CSV |
| `evaluation` | `d_targets` | שיעורי הכישלון שבהם "קוראים" את `T` מעקומת ה-Pareto (`[0.01, 0.05, 0.10]`). רק לניתוח (t8); לא משפיע על הריצה ולא על `config_hash` |

**למה דווקא הערכים האלה?** ראו `docs/decisions/t1_defaults.md`. בקצרה: היתקעות (`k=0`) אף פעם לא גורמת להתנגשות ולכן מותר לה להיות נפוצה, ושגיאות שמתנגשות בקיר (סחיפה, `k=2`) חייבות להיות נדירות.
לבדיקת רגישות עם תנועה גרועה יש preset: `configs/stress.yaml` (`p_steps = [0.1, 0.898, 0.002]`, `p_drift = 0.002`).

כל ערך נבדק (בודקים משותפים `check_float` / `check_int` / `check_bool` / `check_choice`, שגם `policies.py` משתמש בהם):
הסתברויות בין 0 ל-1, `p_steps` חייב להסתכם ל-1, `wall_mode` רק מתוך הרשימה, וכו'.
**מפתח לא מוכר הוא שגיאה, וגם מפתח חסר** — טעות הקלדה לא נופלת בשקט לערך ברירת מחדל. פלט אמיתי:

```
ERROR runner: config error: unknown key 'p_drfit' in motion. Did you mean 'p_drift'? Valid keys: cell_size_mm, diag_rule, ...
ERROR runner: config error: motion.p_drift must be <= 1.0, got 1.5
ERROR runner: config error: motion.extra_stay + motion.extra_over must be <= 0.95, got 1.0
ERROR runner: config error: projecta.tau_lost (0.3) must be below projecta.tau_loc (0.25) so the localize/travel switch has hysteresis
```

### איך משנים פרמטר

ההגדרה הסופית נבנית בשכבות, וכל שכבה דורסת את הקודמת:

1. `configs/default.yaml` — תמיד נטען ראשון (ב-`load_config` אפשר להחליף אותו בבסיס אחר דרך `base_path=`).
2. `--config my_exp.yaml` — קובץ ניסוי שמכיל **רק** את מה שמשתנה, ומתמזג לעומק על הבסיס. סעיף ריק או `null` (למשל `motion:` לבד) פירושו "בלי שינוי".
3. `--set section.key=value` — אפשר כמה פעמים. הערך מפוענח כ-YAML, אז `0.2`, `null`, `[0.1, 0.8, 0.1]` עובדים.
4. הדגלים הייעודיים `--n-runs`, `--seed0`, `--out`, `--workers`, `--policy` — מנצחים אפילו את `--set`.

אחרי המיזוג חייב להיות ערך לכל מפתח. מי שבונה `SimConfig.from_dict` ישירות (בלי מיזוג) חייב לתת הגדרה מלאה.

### `config_hash` — "תעודת זהות" לניסוי

SHA-256 (16 התווים הראשונים) של ההגדרה הסופית **ושל `maze_hash`** (תוכן המבוך שבאמת רץ: קירות ותאי מטרה),
**בלי** הסעיפים `run` ו-`evaluation`, ובלי הפרמטרים של מדיניות שלא נבחרה.
הרעיון: `n_runs`, `seed0`, `workers`, `out` ו-`d_targets` לא משנים *מהו* הניסוי, רק כמה ממנו הרצנו, איפה שמרנו ואיך ננתח. אבל אם קובץ המבוך השתנה — זה כבר ניסוי אחר.

```
default                                 -> f4518eebc6985443
default + run.n_runs=5                  -> f4518eebc6985443   (אותו ניסוי)
default + evaluation.d_targets=[0.02,0.1] -> f4518eebc6985443 (אותו ניסוי)
default + sensing.p_miss=0.1            -> 14892458d0b084d4   (ניסוי אחר)
default + maze.goal=[[0,0]]             -> 2b5d5307e838bd7c   (ניסוי אחר)
```

בקוד: `SimConfig.config_hash(maze_hash=None)` היא מתודה; בלי `maze_hash` היא מחשבת hash של ההגדרה בלבד (ערך אחר!).
ה-runner תמיד משתמש ב-`experiment_hash(cfg, maze)`, שמעביר את `maze_hash` של המבוך שרץ. `effective_config` מוודא ש-`maze.name`
בהגדרה הוא שם המבוך שבאמת נטען (משנה משהו רק כשמבוך מועבר ישירות, למשל בטסטים).

בניתוח (t8) מקבצים שורות לפי `config_hash`, גם אם הגיעו מקבצי CSV שונים. בנוסף נכתב ליד ה-CSV קובץ
`<שם>.config.yaml`: בראשו הערות עם שדות המבוך, ואחריהן ההגדרה המלאה. אפשר להעביר אותו בחזרה ל-`--config` ולקבל אותו `config_hash` (כל עוד המבוך לא השתנה):

```
# Resolved config for default.csv
# config_hash: f4518eebc6985443   (config + maze content)
# maze_name: medium/m10_a
# maze_size: 10x10
# maze_goal_cells: [[4, 4], [4, 5], [5, 4], [5, 5]]
# maze_hash: 6708975ecfe17c79
# motion.p_steps resolved: [0.05000000093715962, 0.9498999981256807, 0.00010000093715965665]
maze: {name: medium/m10_a, goal: null}
...
```

## 4. שחזוריות: אותו seed, אותה תוצאה

לכל אפיזודה יש seed משלה: `seed0, seed0+1, ..., seed0+n_runs-1`. כל האקראיות נגזרת ממנו, בשני **זרמים נפרדים**:

- **זרם הסביבה** — `np.random.default_rng(seed)`. ממנו מוגרלים תא ההתחלה (אחיד על כל התאים שאינם מטרה) ואחר כך, בכל תזוזה, הסחיפה ו-`k`.
- **זרם המדיניות** — `default_rng(SeedSequence(seed).spawn(1)[0])` (הפונקציה `policy_rng`). זרם בלתי תלוי שנגזר מאותו seed.

למה להפריד? אם המדיניות הייתה שולפת מספרים מאותו זרם של הסביבה, מדיניות שמגרילה הרבה (כמו `random`, או POMCP בעתיד)
הייתה "מזיזה" את הזרם — ואז שתי מדיניות היו מתחילות מתאים שונים, וכל השוואה הייתה מערבבת "מדיניות טובה יותר" עם "מזל אחר".
עם ההפרדה, לכל seed **אותו תא התחלה ואותם מספרים אקראיים לרעש התנועה**, לא משנה איזו מדיניות רצה (common random numbers).

**אנלוגיה:** משווים שני נהגים על אותו מסלול. מזג האוויר והמהמורות נקבעים מראש לפי מספר המקצה (זרם הסביבה),
ולכל נהג יש מטבע פרטי להחלטות שלו (זרם המדיניות). אם הנהג היה מטיל את המטבע מאותה קופה של מזג האוויר — כל הטלה שלו הייתה משנה את מזג האוויר.

בדקנו בפועל: ב-1,000 seeds, `random` ו-`oracle_bfs` התחילו מאותם תאים בדיוק; ו-CSV עם `--workers 1` זהה ל-`--workers 4`
(השורות ממוינות לפי seed), חוץ מהעמודה `wall_time_ms` שמודדת זמן אמיתי.
שימו לב: "אותו רעש" פירושו אותו רצף מספרים. מדיניות שבוחרת פעולות אחרות תגיע כמובן למקומות אחרים.

## 5. מדיניות (policies)

### הממשק

```python
policy = make_policy("random", {"declare_after": 20})
policy.reset(env_info, rng)     # בתחילת כל אפיזודה
action = policy.act(obs, info)  # בכל צעד, עד שהאפיזודה נגמרת
policy.episode_stats()          # מדדים אופציונליים ל-CSV
```

- `env_info` (`EnvInfo`) — מה שהרובוט רשאי לדעת מראש: המפה (`wall_map`), תאי המטרה, גודל, מודל הרעש, `max_steps`. **לא** את המיקום.
- הפעולות: 8 כיוונים (`N, NE, E, ..., NW`) ועוד `DECLARE` ("אני במטרה").
- `obs` — התצפית מהחיישנים. עד t4 היא תמיד `None`.
- `info` — **מסונן על ידי ה-runner**: מדיניות רגילה מקבלת רק `t` (כמה פעולות עברו) ו-`last_action`.
  רק מדיניות עם `privileged = True` מקבלת את `info["cell"]` — התא האמיתי. כך אלגוריתם "אמיתי" לא יכול לרמות בטעות (יש על זה טסט).

### שתי מדיניות זמניות

| שם | מה עושה | תפקיד |
|---|---|---|
| `random` | בוחרת כיוון אקראי (מ-8, או מ-4 אם `diagonal: false`), ואחרי `declare_after` תזוזות מכריזה | **רצפה**: מה קורה בלי לחשוב בכלל |
| `oracle_bfs` | **privileged**: יודעת איפה היא. מחשבת BFS מתאי המטרה, בכל צעד הולכת לשכן החוקי הכי קרוב למטרה, ומכריזה רק כשהיא בתא מטרה | **נקודת ייחוס** (reference baseline): כמה שווה לדעת את המיקום. מתכננת כאילו אין רעש, ולכן סחיפה או `k=2` עדיין יכולות לגרום לה להתנגש (נדיר עם ברירות המחדל: 3 מתוך 10,000 על `m10_a`). **זו לא תקרה** — מדיניות שמתחשבת ברעש יכולה להתנגש פחות |

ה-BFS מחושב פעם אחת ונשמר במטמון לפי **תוכן** המבוך (קירות, מטרות, קבוצת הכיוונים), כך שאותה מדיניות יכולה לרוץ בבטחה על כמה מבוכים.

למה הן קיימות לפני האלגוריתם האמיתי (t7)? כדי לבדוק את כל הצינור — config → הרצה → CSV — מקצה לקצה עכשיו,
ולקבל שתי נקודות ייחוס לגרפים. למשל, הטסט בודק שבלי רעש האורקל מצליח ב-100% ו-`T` שווה בדיוק למרחק ה-BFS.

תוצאות אמיתיות (1,000 אפיזודות, `medium/m10_a`):

| הרצה | success | collision | wrong_declaration | ממוצע `T` |
|---|---|---|---|---|
| `oracle_bfs`, ברירת מחדל | 1000 | 0 | 0 | 5.09 |
| `oracle_bfs`, ברירת מחדל, 10,000 אפיזודות | 9997 | 3 | 0 | 5.13 |
| `oracle_bfs`, `--config configs/stress.yaml` | 983 | 17 | 0 | 5.30 |
| `random`, ברירת מחדל | 0 | 1000 | 0 | 1.89 |
| `random`, `motion.wall_mode=stop` | 32 | 0 | 968 | 20.00 |

עם ברירות המחדל האורקל כמעט לא נכשל (0.03% ב-10,000 ריצות; ב-`t1_defaults.md` "רצפת הכישלון" המחושבת ל-10×10 היא כ-0.1%), אז הכישלונות שיראו אלגוריתמים אמיתיים יבואו מאי-ודאות במיקום ולא מרעש התנועה.
ב-`stress.yaml` הרעש הגס גדול פי 20, ורואים את זה מיד (1.7%). שימו לב שגם בהצלחה, `T` יכול להיות גדול מהמרחק הקצר ביותר: ב-5% מהצעדים הרובוט "נתקע" (`k=0`) וצריך לחזור על הצעד.

**להוסיף מדיניות חדשה:** לרשת מ-`Policy`, לקבוע `name` ו-`DEFAULTS` (שמות הפרמטרים), לממש `act`, להוסיף למילון `POLICIES`,
ולהוסיף כניסה עם **כל** הפרמטרים תחת `policy.params` ב-`default.yaml`. בהרצה מ-config הערכים באים רק מה-YAML;
`DEFAULTS` משמש למילוי רק כשבונים מדיניות ישירות בקוד.

## 6. ה-CSV: עמודה אחר עמודה

| עמודה | משמעות |
|---|---|
| `run_id` | `<config_hash>-<seed>` — מזהה ייחודי לאפיזודה |
| `seed` | ה-seed של האפיזודה |
| `maze`, `maze_hash` | שם המבוך, ו-hash של תוכנו (גודל, קירות, תאי מטרה). אם קובץ מבוך ישתנה — `maze_hash` ישתנה, וגם `config_hash` |
| `policy`, `policy_params_json` | שם המדיניות והפרמטרים שלה, כ-JSON |
| `T` | העלות: `n_moves × step_cost`, ועוד `declare_cost` אם הייתה הכרזה. התזוזה שהתנגשה נספרת |
| `outcome` | `success` / `wrong_declaration` / `collision` / `timeout` |
| `failed` | `1` לכל מה שאינו `success`, אחרת `0` |
| `n_moves` | מספר התזוזות (בלי ה-`DECLARE`) |
| `steps_to_loc`, `final_P_goal`, `n_fallbacks` | מדדים שהמדיניות מחזירה ב-`episode_stats()`. **ריקים כרגע** (יתמלאו ב-t7) |
| `start_r`, `start_c` | תא ההתחלה (האמיתי) |
| `final_r`, `final_c` | התא בסוף. בהתנגשות: התא האחרון לפני הקיר |
| `collision_action`, `collision_direction`, `collision_k` | בהתנגשות: הכיוון שנצטווה, הכיוון שבפועל (אחרי סחיפה), ו-`k` שהוגרל. ריק אם לא הייתה התנגשות |
| `motion.*`, `sensing.*`, `episode.*`, `projecta.*` | כל פרמטרי הניסוי, עמודה לכל אחד (כולל `motion.cell_size_mm`). `motion.p_steps` ריק כשהוא `null`; `motion.p_step0/1/2` הם הערכים שבאמת שימשו. `evaluation` לא נכתב ל-CSV |
| `wall_time_ms` | כמה זמן לקחה האפיזודה. העמודה היחידה שמשתנה בין הרצות זהות (השורה הראשונה איטית יותר כי ה-BFS מחושב בה) |
| `config_hash` | ראו סעיף 3 (הגדרה + תוכן המבוך) |

## 7. איך מריצים

מתוך התיקייה `lidar_sim`. ב-PowerShell:

```powershell
cd "C:\Users\segev\Documents\Technion\Semester_9\Project B\lidar_sim"

# 1,000 אפיזודות עם ברירת המחדל -> results\default.csv + results\default.config.yaml
..\.venv\Scripts\python.exe runner.py

# מדיניות אחרת, קובץ פלט אחר
..\.venv\Scripts\python.exe runner.py --policy random --set motion.wall_mode=stop --out results/random_stop.csv

# ה-preset של תנועה גרועה (כותב ל-results/stress.csv)
..\.venv\Scripts\python.exe runner.py --config configs/stress.yaml

# שינוי פרמטר בודד (ערך שהוא רשימה צריך גרשיים)
..\.venv\Scripts\python.exe runner.py --set motion.p_drift=0.001 --set "motion.p_steps=[0.05, 0.949, 0.001]" --out results/drift.csv

# מבוך אחר, 200 אפיזודות, החל מ-seed 1000
..\.venv\Scripts\python.exe runner.py --set maze.name=small/m5_a --n-runs 200 --seed0 1000 --out results/m5.csv

# קובץ ניסוי (מכיל רק את מה שמשתנה), ו-4 תהליכים במקביל
..\.venv\Scripts\python.exe runner.py --config configs/my_exp.yaml --workers 4

# הטסטים
..\.venv\Scripts\python.exe -m pytest test_runner.py -q
```

פלט אמיתי של ההרצה הראשונה (בקיצור):

```
INFO runner: running 1000 episodes: maze=medium/m10_a (10x10) policy=oracle_bfs workers=1 config_hash=f4518eebc6985443
INFO runner: 100/1000 episodes, 0.0 s
...
INFO runner: finished 1000 episodes in 0.12 s
INFO runner: outcomes: {'success': 1000, 'wrong_declaration': 0, 'collision': 0, 'timeout': 0}
INFO runner: failure rate 0.0000 (0 of 1000), mean T 5.09 (successes 5.09)
INFO runner: wrote ...\default.csv and ...\default.config.yaml
```

כמה שורות מה-CSV (רק חלק מהעמודות; בקובץ האמיתי יש 48):

```
run_id,seed,policy,T,outcome,failed,n_moves,start_r,start_c,final_r,final_c,collision_action,motion.p_step0,motion.p_drift,wall_time_ms
f4518eebc6985443-0,0,oracle_bfs,5.0,success,0,5,8,5,5,5,,0.05000000093715962,0.0001,14.264
f4518eebc6985443-1,1,oracle_bfs,3.0,success,0,3,4,7,5,5,,0.05000000093715962,0.0001,0.126
f4518eebc6985443-2,2,oracle_bfs,5.0,success,0,5,8,4,5,5,,0.05000000093715962,0.0001,0.145
f4518eebc6985443-3,3,oracle_bfs,6.0,success,0,6,8,1,4,4,,0.05000000093715962,0.0001,0.145
```

(בשורה 1: מתא `(8,5)` המרחק הקצר ביותר הוא 4 צעדים, אבל `n_moves = 5` — צעד אחד "נתקע" (`k=0`) וחזר על עצמו.
`collision_action` ריק כי לא הייתה התנגשות.)

והטסטים: `53 passed in 1.59s`.

יעד t5.2 (1,000 אפיזודות על 10×10 בפחות מ-10 דקות) מתקיים בקלות כרגע.

> **דברים שכדאי לדעת**
> - `random` במצב `wall_mode: collide` נכנסת בקיר כמעט תמיד (1000 מתוך 1000). היא שימושית רק עם `--set motion.wall_mode=stop`.
> - הדגלים הייעודיים (`--n-runs`, `--seed0`, `--out`, `--workers`, `--policy`) מנצחים את `--set`: `--set run.n_runs=5 --n-runs 10` ירוץ 10.
> - `run.out` יחסי **לתיקייה הנוכחית**, לא ל-`lidar_sim`. הרצה משורש הפרויקט תיצור `results/` שם.
> - `--workers` גדול מ-1 איטי יותר היום (0.43 שנ' מול 0.12 ל-1,000 אפיזודות), כי הקמת התהליכים יקרה מהאפיזודות עצמן. זה ישתלם עם מדיניות כבדה (t7).

## 8. מה עוד אין כאן

- **חיישנים (t4):** `obs` תמיד `None`, וסעיף `sensing` נבדק ונכנס ל-hash ול-CSV אבל לא משפיע על כלום.
- **מדיניות אמיתית (t7):** היוריסטיקת פרויקט A עם belief. סעיף `projecta` מחכה לה; גם הוא לא משפיע כרגע.
- **עמודות ריקות:** `steps_to_loc`, `final_P_goal`, `n_fallbacks`.
- **סעיפי `pomdp` ו-`mujoco`** שמופיעים ב-ROADMAP §1.2 עוד לא קיימים בקובץ (כרגע הם היו נדחים כמפתח לא מוכר).
- **ניתוח וגרפים (t8):** ה-CSV מוכן לזה, הקוד עוד לא.

## מילון קצר

| מונח | פירוש |
|---|---|
| אפיזודה | ריצה אחת: חטיפה לתא אקראי, ניסיון להגיע למטרה, עד אחת מארבע התוצאות |
| headless | ריצה בלי חלון גרפי (בלי pygame) |
| seed | מספר שקובע את כל האקראיות של אפיזודה |
| RNG / `Generator` | מחולל מספרים אקראיים של numpy |
| common random numbers | אותם מספרים אקראיים לכל המדיניות, כדי שההשוואה תהיה הוגנת |
| `config_hash` | טביעת אצבע של הניסוי: ההגדרות (בלי `run`) + תוכן המבוך |
| `maze_hash` | טביעת אצבע של המבוך עצמו: גודל, קירות, תאי מטרה |
| sidecar | קובץ `.config.yaml` שנכתב ליד ה-CSV עם ההגדרה המלאה ופרטי המבוך |
| privileged | מדיניות שמותר לה לראות את המיקום האמיתי; רק לייחוס |
| reference baseline | נקודת ייחוס להשוואה (לא בהכרח הטוב ביותר האפשרי) |
| worker | תהליך מקבילי שמריץ חלק מה-seeds |
| `DECLARE` | הפעולה "אני במטרה" — מסיימת את האפיזודה |
| `T` | עלות האפיזודה, בעיקר מספר התזוזות |
