## Прунинг LaBSE-en-ru

Прунинг - это удаление определенных частей модели (подрезание)

Почему пруним большую сеть, а не берем сетку поменьше. Есть работы, показывающие, что пруненная сеть выигрывает
по latency или по accuracy у сеток, аналогичного размера

Прунинг можно воспринимать как правильную инициализацию сети ученика. 

**Мы подрежем оригинальную сеть и будем её обучать, стягивая выходы слоев запруненной и оригинальной сети** 


### Данные

Для обучения используется корпус парных предложений https://translate.yandex.ru/corpus. 

### Настройка окружения

1. Создание
    ```
    python3 -m venv /path/to/new/virtual/environment
    source /path/to/new/virtual/environment/bin/activate
    ```

2. Установка зависимостей

    ```
    pip install -r requirements.txt
   
    git clone https://github.com/avidale/encodechka
    python encodechka/setup.py install
    rm -rf encodechka
    ```

3. Настройка ClearML

   - Регистрируемся в [ClearML](https://app.community.clear.ml/), если ещё нет аккаунта.
   - [в своем профиле ClearML](https://app.community.clear.ml/profile) нажимаем "Create new credentials"
   - в консоли пишем `clearml-init` и следуем инструкциям


### Запуск пайплайна
```
make train
```

### Просмотр экспериментов
   -  Локально в папке **experiments** лежат чекпойнты лучших моделей
   -  В [ClearML](https://app.clear.ml/projects/8a15381deb0d41429e451070a014c1a3)

### Проверка качества эмбеддингов на бейчмарке encedechka
```
make infer - для запуска в терминальном режиме
notebooks/encodechka_evaluation_2024.ipynb - jupyter блокнот 
```
