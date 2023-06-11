import os
import typing
import tempfile
from bitstring import ConstBitStream
from huffman import compression

# union few files
# 2-methods support
# uncompress file with custom number / filename
# delete file with custom number / filename
# info about all files in archive


# noinspection PyArgumentList
class VarInt:  # unsigned
    def __init__(self, size: int):
        self.size = size
        self.max_value = 255 - size

    def unpack(self, stream: typing.BinaryIO) -> int:
        value = stream.read(1)[0]

        if value > self.max_value:
            value = int.from_bytes(stream.read(value - self.max_value))

        return value

    def pack(self, integer: int) -> bytes:
        if integer < self.max_value:
            return integer.to_bytes(1)

        length = integer.bit_length()
        size = length // 8 + (length % 8 > 0)

        if size > self.size:
            raise ValueError(f'incorrect value for size {self.size}')

        return (self.max_value + size).to_bytes(1) + integer.to_bytes(size)


class Header:
    """
    -- HEADER --
    <filename size:varint2> <filename:utf8> <uncompressed size:varint6>
    <compressed size:varint6> <offset:1> <method> [<encoding>] -> <payload>
    """
    _fn_size = VarInt(2)
    _uncompressed_size = VarInt(6)
    _compressed_size = VarInt(6)

    def __init__(self, fn: str | bytes, uncompressed_size: int,
                 compressed_size: int, offset: int, method: str,
                 encoding: compression.Encoding = None):

        if method not in ['onepass', 'twopass']:
            raise ValueError(f'unknown method "{method}"')

        self._fn_encoded = fn.encode() if isinstance(fn, str) else fn
        self.fn_size = len(self._fn_encoded)
        self.fn = fn if isinstance(fn, str) else fn.decode()
        self.uncompressed_size = uncompressed_size
        self.compressed_size = compressed_size
        self.offset = offset
        self.method = method
        self.encoding = encoding

    def __repr__(self):
        return f'Header(fn={self.fn}, ' \
               f'uncompressed_size={self.uncompressed_size}, ' \
               f'compressed_size={self.compressed_size}, ' \
               f'offset={self.offset}, ' \
               f'method={self.method}' \
               f'encoding-size={len(self.encoding.pack()) if self.encoding else None}, ' \
               f'full-compressed-size={len(self.pack()) + self.compressed_size})'

    @property
    def size(self) -> int:
        return len(self.pack())

    def get_compressed_size_pos(self) -> int:
        return sum(map(len, [
            self._fn_size.pack(self.fn_size),
            self._fn_encoded,
            self._uncompressed_size.pack(self.uncompressed_size),
        ]))

    # noinspection PyArgumentList
    def pack(self) -> bytes:
        return b''.join((
            self._fn_size.pack(self.fn_size),
            self._fn_encoded,
            self._uncompressed_size.pack(self.uncompressed_size),
            self._compressed_size.pack(self.compressed_size),
            self.offset.to_bytes(1),
            b'\x00' if self.method == 'onepass' else b'\x01',
            b'' if self.method == 'onepass' else self.encoding.pack()
        ))

    # noinspection PyArgumentList
    @classmethod
    def unpack(cls, stream: typing.BinaryIO) -> 'Header':
        return cls(
            stream.read(cls._fn_size.unpack(stream)),
            cls._uncompressed_size.unpack(stream),
            cls._compressed_size.unpack(stream),
            int.from_bytes(stream.read(1)),
            method := 'twopass' if int.from_bytes(stream.read(1)) else 'onepass',
            compression.Encoding.unpack(stream) if method == 'twopass' else None
        )


class Archive:
    def __init__(self, fp: str):
        self.fp = os.path.abspath(fp)
        self.size = os.stat(fp).st_size
        self.headers = self._get_headers()

    def update(self):
        self.__init__(self.fp)

    @property
    def uncompressed_size(self) -> int:
        return sum(h.uncompressed_size for h in self.headers)

    def _get_headers(self) -> list[Header]:
        i, headers, size = 0, [], os.stat(self.fp).st_size

        with open(self.fp, 'rb') as f:
            while f.tell() < size:
                headers.append(header := Header.unpack(f))
                f.seek(f.tell() + header.compressed_size)

        return headers

    def _delete_files(self, files: list[str | int], of: typing.BinaryIO, nf: typing.BinaryIO) -> int:
        filenum, deleted_count = -1, 0
        while of.tell() < self.size:
            h = Header.unpack(of)
            filenum += 1

            block_end = of.tell() + h.compressed_size
            if h.fn in files or filenum in files:
                deleted_count += 1
                of.seek(block_end)
                continue

            nf.write(h.pack())
            while of.tell() < block_end:
                nf.write(of.read(1))

        return deleted_count

    def delete_files(self, files: list[str | int]) -> int:
        tmp_dir = tempfile.mkdtemp()
        tmp = os.path.join(tmp_dir, os.path.basename(self.fp))
        with open(tmp, 'wb') as nf:
            with open(self.fp, 'rb') as of:
                deleted = self._delete_files(files, of, nf)

        os.replace(tmp, self.fp)
        os.rmdir(tmp_dir)
        return deleted

    def _uncompress_payload(self, header: Header, cf: typing.BinaryIO, uf: typing.BinaryIO):
        stream = ConstBitStream(filename=self.fp,
                                length=header.compressed_size * 8 - header.offset,
                                offset=cf.tell() * 8)

        if header.method == 'onepass':
            decoder = compression.OnePassDecoder(stream)

        elif header.method == 'twopass':
            decoder = compression.TwoPassDecoder(header.encoding, stream)

        # noinspection PyUnboundLocalVariable
        for byte in decoder.decode():
            uf.write(byte)

        cf.seek(cf.tell() + header.compressed_size)

    def uncompress(self, output_d: str, files: list[str | int] = None) -> list[Header]:
        if not os.path.exists(output_d):
            os.makedirs(output_d)

        filenum, uncompressed_fns, uncompressed_headers = -1, {}, []
        with open(self.fp, 'rb') as cf:
            while cf.tell() < self.size:
                h = Header.unpack(cf)
                filenum += 1

                if files and all(map(lambda x: x not in files, [filenum, h.fn])):
                    cf.seek(cf.tell() + h.compressed_size)
                    continue

                uncompressed_fns[h.fn] = uncompressed_fns.get(h.fn, 0) + 1
                if uncompressed_fns[h.fn] > 1:
                    name, ext = os.path.splitext(os.path.basename(h.fn))
                    fn = f'{name} ({uncompressed_fns[h.fn]}){ext}'

                else:
                    fn = h.fn

                with open(os.path.join(output_d, fn), 'wb') as uf:
                    self._uncompress_payload(h, cf, uf)

                h.fn = fn
                uncompressed_headers.append(h)

        return uncompressed_headers

    # noinspection PyArgumentList
    @staticmethod
    def _add_file(cf: typing.BinaryIO, uf: typing.BinaryIO, fn: str, uncompressed_size: int, method: str):
        fs = cf.tell()
        encoder = (compression.OnePassEncoder if method == 'onepass' else compression.TwoPassEncoder)(uf)
        header = Header(
            fn,
            uncompressed_size,
            256 ** Header._compressed_size.size - 1 if method == 'onepass' else encoder.encoded_size,
            0 if method == 'onepass' else encoder.encoded_offset,
            method,
            None if method == 'onepass' else encoder.encoding
        )
        cf.write(header.pack())

        for payload in encoder.encode():
            cf.write(payload)

        if method == 'onepass':
            size, offset = encoder.get_size_inf()

            end = cf.tell()
            cf.seek(fs + header.get_compressed_size_pos() + 1)  # + 1 cause VarInt (pass first byte)
            cf.write(size.to_bytes(Header._compressed_size.size))
            cf.write(offset.to_bytes(1))
            cf.seek(end)

    @classmethod
    def create(cls, path: str, paths: list[str], method: str) -> 'Archive':
        with open(path, 'wb') as cf:
            for ph in paths:
                with open(ph, 'rb') as uf:
                    cls._add_file(cf, uf, os.path.basename(ph), os.stat(ph).st_size, method)

        return cls(path)
