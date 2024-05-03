## Прунинг LaBSE_en_ru

Прунинг - это удаление определенных частей модели (подрезание)

Почему пруним большую сеть, а не берем сетку поменьше. Есть работы, показывающие, что пруненная сеть выигрывает
по latency или по accuracy у сеток, аналогичного размера

Прунинг можно воспринимать как правильную инициализацию сети ученика


### Данные

Для обучения используется корпус парных предложений https://translate.yandex.ru/corpus. 

### Подготовка пайплайна

1. Создание и активация окружения
    ```
    python3 -m venv /path/to/new/virtual/environment
    ```
    ```
    source /path/to/new/virtual/environment/bin/activate
    ```

2. Установка пакетов

    В активированном окружении:
    ```
    pip install -r requirements.txt
    ```

3. Настройка ClearML

   - Регистрируемся в [ClearML](https://app.community.clear.ml/), если ещё нет аккаунта.
   - [в своем профиле ClearML](https://app.community.clear.ml/profile) нажимаем "Create new credentials"
   - в консоли пишем `clearml-init` и следуем инструкциям


### Запуск обучения
```
make train
```

### Результаты экспериментов
   -  Локально в папке **experiments** лежат чекпойнты лучших моделей
   -  В [ClearML](https://app.clear.ml/projects/8a15381deb0d41429e451070a014c1a3)

### Сохранение результатов лучшего экперимента в каталоге моделей
```
make save
```

### Проверка качества эмбеддингов на бейчмарке encedechka
```
# Запуск инференса в терминале
make infer
```
```
# Пример инференса в jupyter
notebook/inference.ipynb 
```
