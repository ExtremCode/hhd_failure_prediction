# hdd_failure_prediction

## Введение

С увеличением объемов данных и ростом зависимости бизнеса от информационных технологий, надежность хранения информации становится критически важной. Жесткие диски (HDD) остаются основным средством хранения данных, однако их выход из строя может привести к значительным потерям. Одним из перспективных подходов к предсказанию отказов жестких дисков является анализ S.M.A.R.T (Self-Monitoring, Analysis, and Reporting Technology) [1] показателей, которые предоставляют информацию о состоянии устройства и его производительности. Актуальность данной задачи обусловлена необходимостью разработки более эффективных алгоритмов, способных на ранних стадиях выявлять потенциальные проблемы, что позволит минимизировать риски потери данных и снизить затраты на обслуживание. В данной работе рассматриваются современные методы анализа SMART показателей и их применение для предсказания выхода из строя жестких дисков.

## Постановка задачи

Учитывая данные мониторинга состояния диска S.M.A.R.T и неисправностях, необходимо определить, выйдет ли из строя каждый диск в течение следующих 30 дней. Качество классификации оценивалось с помощью метрики *ROC AUC* (англ. Area Under the Receiver Operating Characteristic Curve).

## Предварительный анализ данных

Предоставленные данные содержат ненормированные значения S.M.A.R.T показателей жестких дисков модели ST14000NM001G. Общее число записей 7320142. Частота дискретизации измерений составляет 1 день. В случае выхода из строя диска в его последнем дне в столбце `failure` стоит 1. Зафиксированное число дней работы дисков составляет от 1 до 828.

Отсутсвуют пропущенные значения и дубликаты. Все признаки имеют верные типы данных, таблица 1.

Таблица 1. Названия признаков и типы данных
 номер |  Признка  |   Тип данных     
--- | ------     |     -----         
 0  | date          |  datetime64
 1  | serial_number  | object        
 2  | model           |object        
 3  | capacity_bytes  |int64         
 4  | failure        | int64         
 5  | smart_5_raw    | float64       
 6  | smart_9_raw    | float64       
 7  | smart_187_raw  | float64       
 8  | smart_188_raw  | float64       
 9  | smart_192_raw  | float64       
 10 | smart_197_raw  | float64       
 11 | smart_198_raw  | float64       
 12 | smart_199_raw  | float64       
 13 | smart_240_raw  | float64       
 14 | smart_241_raw  | float64       
 15 | smart_242_raw  | float64

Признаки `capacity_bytes` и `model` имеют только одно уникальное значение, в дальнейшем они не будут учитываться.

Число вышедших из строя жестких дисков составляет 172. Это свидетельствует о дисбалансе классов, который объясняется тем, что выход из строя диска не является частой и обычной ситуацией. Минимальное число дней, которое проработал диск, вышедший из строя – 12, 25-ый процентиль составляет 306 дней.

## Преобразование данных и новые признаки

Для построения обучающего набора данных были выбраны последние 41 день работы диска. Из-за этого в набор не вошли 16 дисков. Для дисков, вышедших из строя, во всех строках признак `failure` был равен 1.

На основе этих данных отдельно для каждого диска вычислялись новые признаки для каждого S.M.A.R.T показателя:
* среднее по 5, 7 и 10 дням экспоненциально взвешенных абсолютных разниц между следующим значением и предыдущим
* сдвиг значений из прошлого в будущее (лаг) на 5, 7 и 10 дней

Удалялись строки, содержащие хотя бы один пропущенный элемент в новых признаках. Таким образом был составлен набор данных, включающий 31 день измерений для каждого диска. Так как показатели имеют разные дни записи, то для каждого диска был проставлен номер дня, начиная с 1. Этот номер позволяет создать универсальную общую временную шкалу.

## Выбор предпочтительных признаков

Выбор признаков осуществляся на основе:
* статистических тестов
* на основании результатов похожих работ [2]–[5]
* используя свойства метрики L1 в обучении различных моделей

### Выбор по статистическим данным

Для определения информативных признаков были ипользованы критические значения теста Фишера и взаимной информации. Чем выше была величина, тем более значимым является признак. Исходя из этого, были выбраны признаки, имеющие нормированные критические значения теста Фишера или взаимной информации больше 0,7.

Также обнаружилась высокая корреляция `smart_198_raw` с `smart_197_raw` также `smart_240_raw` с `smart_9_raw`, `smart_241_raw` и `smart_242_raw`. Соответственно был рассмотрен вариант удаления коррелированных признаков.

### Выбор на основании схожих работ

При решении аналогичной задачи [2] отмечена информативность признаков `smart_5_raw`, `smart_187_raw`, `smart_188_raw`, `smart_197_raw` и `smart_198_raw`.

### Выбор, используя обученные модели

Для выбора признаков производилось обучение логистической регрессии с использованием одновременно регуляризаций L1 и L2. Наиболее предпочтительные признаки имели ненулевые коэффициенты в модели.

Также была обучена модель градиентного бустинга над решающими деревьями с использованием регуляризации L1. На основе построенной модели были выбраны 4 группы признаков, каждая из которых составляет треть от общего числа всех признаков:
* которые участвовали в наибольшем числе разбиений
* привнесли наибольший средний прирост информации
* привнесли наибольший суммарный прирост информации
* участвовали при разделении наибольшего среднего количества примеров

## Уменьшение размерности

Уменьшение размерности производилось с помощью выделения 20 главных компонент, используя анализ главных компонент, факторный анализ и матричную факторизацию.

## Балансировка классов

Рассматривались различные способы балансировки классов, такие как:
* SMOTE
* KMeansSMOTE
* ADASYN
* использование весовых коэффициентов

## Обучение и валидация моделей

Основываясь на соответствующих работах [6]–[9], были применены модели:
* градиентный бустинг над решающими деревьями (XGBClassifier)
* случайный лес (RandomForest)
* изолирующий лес (IsolationForest)
* сбалансированный случайный лес (BalancedRandomForest)
* блендинг моделей (Blending)

Значения метрики *ROC AUC* для обученных моделей для валидационной выборки представлены в таблице 2.

Таблица 2. Валидационное значение метрики *ROC AUC* различных моделей
Название | *ROC AUC*
--- | ---
XGBClassifier | 0,99520
RandomForest | 0,99167
IsolationForest | 0,88579
BalancedRandomForest | 0,95749
Blending | 0,99285

## Список использованных источников
[1] S.M.A.R.T. [Электронный ресурс]. – Режим доступа: https://ru.wikipedia.org/wiki/S.M.A.R.T. (дата обращения: 20.12.2024)

[2] KarthikNA. Prediction of Hard Drive Failure [Электронный ресурс]. – Режим доступа: https://github.com/KarthikNA/Prediction-of-Hard-Drive-Failure (дата обращения:  20.12.2024)

[3] Tianchi. Информация о конкурсе [Электронный ресурс]. – Режим доступа: https://tianchi.aliyun.com/competition/entrance/231775/information (дата обращения: 20.12.2024)

[4] Harish Kumar. HDD Failure Detection [Электронный ресурс]. – Режим доступа: https://harishkumar-69065.medium.com/hdd-failure-detection-4a4797fae7e (дата обращения: 20.12.2024)

[5] Scot Comp. Preemptive Measures: Predicting Hard Drive Failures [Электронный ресурс]. – Режим доступа: https://scotcomp.medium.com/preemptive-measures-predicting-hard-drive-failures-01210c7e00a5 (дата обращения: 20.12.2024)

[6] Li Q., Li H., Zhang K. Prediction of HDD failures by ensemble learning //2019 IEEE 10th International Conference on Software Engineering and Service Science (ICSESS). – IEEE, 2019. – С. 237-240.

[7] Kodatsky N. et al. Using machine learning to forecast hard drive failures //E3S Web of Conferences. – EDP Sciences, 2024. – Т. 549. – С. 08024.

[8] Zhao J. et al. Disk failure early warning based on the characteristics of customized smart //2020 19th IEEE Intersociety Conference on Thermal and Thermomechanical Phenomena in Electronic Systems (ITherm). – IEEE, 2020. – С. 1282-1288.

[9] Zhang T., Wang E., Zhang D. Predicting failures in hard drivers based on isolation forest algorithm using sliding window //Journal of Physics: Conference Series. – IOP Publishing, 2019. – Т. 1187. – №. 4. – С. 042084.
