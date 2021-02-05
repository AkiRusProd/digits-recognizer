# DigitRecognizer

### Полносвязная нейронная сеть, которая распознает числа от 0 до 9

## Скрипты

`Neuron_training.py`- скрипт, в котором происходит обучение сети. Обученные веса csv формата сохраняются в папке "weights".
(Постскриптум, веса уже готовы, ничего обучать не надо. Но если хочется, то нужно скачать MNIST датасет по ссылке: "https://pjreddie.com/projects/mnist-in-csv/"
,закинуть в папку "dataset",запустить`Neuron_training.py` и подождать некоторое время)

`main.py` - обученная сеть с интерфейсом, которая берет готовые веса и распознаёт числа. 

## Зависимости
* numpy   v1.19.3
* Pillow  v8.0.1
* tkinter