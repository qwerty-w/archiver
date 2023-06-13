import os
import time
import typing
import argparse

from huffman.archive import Archive, Header


parser = argparse.ArgumentParser('Archiver')
parser.add_argument('key',
                    choices=['a', 'x', 'l', 'd'],
                    help='Ключ для операции с архивом: x - извлечение, a - создание нового архива, '
                         'l - список файлов, d - удаление файла из архива')
parser.add_argument('paths',
                    nargs='+',
                    help=' / '.join([
                        'Путь к архиву при создании (a) и пути к файлам для создания архива',
                        'Путь к архиву при извлечении (x)',
                        'Путь к архиву при получении просмотре списка файлов (l)',
                        'Путь к архиву при удалении файлов (d)']))
parser.add_argument('-o',
                    '--output',
                    default=os.getcwd(),
                    help='Путь к папке для извлечения содержимого архива (только для ключа x)')
parser.add_argument('--method',
                    choices=['onepass', 'twopass'],
                    default='twopass',
                    help='Выбор алгоритма для прохода по данным при создании архива')
parser.add_argument('--by-indexes',
                    nargs='+',
                    default=[],
                    type=int,
                    help='Список порядковых номеров файлов для извлечения или удаления из архива')
parser.add_argument('--by-names',
                    nargs='+',
                    default=[],
                    help='Список имен файлов для извлечения или удаления из архива')


def time_wrap(func, *args, _round: int = 2, **kwargs) -> tuple[float, typing.Any]:
    t0 = time.time()
    rv = func(*args, **kwargs)
    return int((time.time() - t0) * 10 ** _round) / 10 ** _round, rv


class Printer:
    @staticmethod
    def create(_time: float, method: str, path: str, files_count: int, uncompressed_size: int, compressed_size: int):
        print(f'Successful create:\n'
              f'Time - {_time}s\n'
              f'Method - {method}\n'
              f'Archive path - {path}\n'
              f'Files added - {files_count}\n'
              f'Uncompressed size - {uncompressed_size}\n'
              f'Compressed size - {compressed_size}\n'
              f'Ratio - {int(uncompressed_size / compressed_size * 100) / 100}')

    @staticmethod
    def extract(_time: float, method: str, output_dir: str, total_files: int, uncompressed_size: int):
        print(f'Successful extract:\n'
              f'Time - {_time}s\n'
              f'Method - {method}\n'
              f'Output dir - {output_dir}\n'
              f'Total files - {total_files}\n'
              f'Uncompressed size - {uncompressed_size}')

    @staticmethod
    def listing(path: str, uncompressed_size: int, compressed_size: int, files: list[Header]):
        print(f'Listing archive: {path}\n'
              f'Total files - {len(files)}\n'
              f'Compressed size - {compressed_size}\n'
              f'Uncompressed size - {uncompressed_size}\n'
              f'\nName [size]\n'
              f'{"-" * 20 + chr(10)}'
              f'{chr(10).join(f"{i}. {f.fn} [{f.uncompressed_size}]" for i, f in enumerate(files))}')

    @staticmethod
    def delete(_time: float, count: int, names: list[str], indexes: list[int],
               size_before: int, size_after: int, uncompressed_size: int):
        print(f'Successful delete {count} files:\n'
              f'Time - {_time}s\n'
              f'Names to delete - {repr(names)}\n'
              f'Indexes to delete - {repr(indexes)}\n'
              f'Compressed size before/after - {size_before}/{size_after}\n'
              f'Uncompressed size - {uncompressed_size}')


def main():
    args = parser.parse_args()
    key, paths = args.key, args.paths
    fp = paths[0]

    if key != 'a' and not os.path.exists(fp):
        parser.error(f'archive {fp} not found')

    match key:
        case 'a':
            files = paths[1:]

            if not files:
                parser.error('files for compression not specified')

            for p in files:
                if not os.path.exists(p):
                    parser.error(f"file {p} not found")

            t, a = time_wrap(Archive.create, fp, files, args.method)
            Printer.create(t,
                           args.method,
                           fp,
                           len(files),
                           a.uncompressed_size,
                           a.size)

        case 'x':
            a, fs = Archive(fp), args.by_indexes + args.by_names
            t, headers = time_wrap(a.uncompress, args.output, fs if fs else None)
            method = headers[0].method if headers else 'huffman'
            Printer.extract(t, method, args.output, len(headers), sum(h.uncompressed_size for h in headers))

        case 'l':
            a = Archive(fp)
            Printer.listing(fp, a.uncompressed_size, a.size, a.headers)

        case 'd':
            a = Archive(fp)
            size_before = a.size
            fs = args.by_indexes + args.by_names

            if not fs:
                parser.error('--by-names/--by-indexes not specified')

            t, count = time_wrap(a.delete_files, fs)
            a.update()
            Printer.delete(t, count, args.by_names, args.by_indexes, size_before, a.size, a.uncompressed_size)


if __name__ == '__main__':
    main()
