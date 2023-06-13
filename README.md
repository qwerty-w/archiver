# Coursework: Archiver 
### Support methods:
|   Key   |          Description          | Speed | Level |
|:-------:|:-----------------------------:|:-----:|:-----:|
| onepass | _FGK Adaptive Huffman Coding_ |   -   |  ++   |
| twopass |   _Classic Huffman Coding_    |   +   |   +   |
### Usage:
```commandline
Archiver [-h] [-o OUTPUT] [--method {onepass,twopass}] [--by-indexes BY_INDEXES [BY_INDEXES ...]] [--by-names BY_NAMES [BY_NAMES ...]] {a,x,l,d} paths [paths ...]

positional arguments:
  {a,x,l,d}             Ключ для операции с архивом: x - извлечение, a - создание нового архива, l - список файлов, d - удаление файла из архива
  paths                 Путь к архиву при создании (a) и пути к файлам для создания архива / Путь к архиву при извлечении (x) / Путь к архиву при получении просмотре списка файлов (l) / Путь к архиву при удалении файлов (d)

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Путь к папке для извлечения содержимого архива (только для ключа x)
  --method {onepass,twopass}
                        Выбор алгоритма для прохода по данным при создании архива
  --by-indexes BY_INDEXES [BY_INDEXES ...]
                        Список порядковых номеров файлов для извлечения или удаления из архива
  --by-names BY_NAMES [BY_NAMES ...]
                        Список имен файлов для извлечения или удаления из архива
```
### Examples:
<details>
  <summary>Create from .txt books</summary>
  <img src="https://i.imgur.com/vbtY5lC.png">
</details>
<details>
  <summary>Onepass method from .txt books</summary>
  <img src="https://i.imgur.com/7rtFQmm.png">
</details>
<details>
  <summary>Delete from archive</summary>
  <img src="https://i.imgur.com/s0AaMaU.png">
</details>
<details>
  <summary>List archive, extract by name</summary>
  <img src="https://i.imgur.com/H9Mtvj3.png">
</details>
<details>
  <summary>Create from pictures (compressed formats: webp, jpg)</summary>
  <img src="https://i.imgur.com/R3Jt4Dh.png">
</details>