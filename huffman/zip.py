import os
import struct
import typing
from huffman import ConstBitStream, BitArray, compression

# create few files -> union
# decode file with custom number
# delete file with custom number
# info about all files in zi


# -- HEADER --
# <filename size:varint> <filename> <uncompressed size:varint> <compressed size:varint> <offset:1> <encoding>
# ENCODING (in bytes): <byte><0/1><0/1><0/1><\n> ... <byte><0/1><0/1><0/1...>


# _is_longest_T = typing.TypeVar()  # todo:


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
            raise ValueError(f'incorrect value for {self.size} size')

        return (self.max_value + size).to_bytes(1) + integer.to_bytes(size)


class Header:
    _fn_size = VarInt(2)
    _uncompressed_size = VarInt(6)
    _compressed_size = VarInt(6)

    def __init__(self, fn: str | bytes, uncompressed_size: int,
                 compressed_size: int, offset: int,
                 encoding: dict[bytes, BitArray]):

        self._fn_encoded = fn.encode() if isinstance(fn, str) else fn

        self.fn_size = len(self._fn_encoded)
        self.fn = fn if isinstance(fn, str) else fn.decode()
        self.uncompressed_size = uncompressed_size
        self.compressed_size = compressed_size
        self.offset = offset
        self.encoding = encoding

    def get_offset_position(self):
        return sum(map(len, [
            self._fn_size.pack(self.fn_size),
            self._fn_encoded,
            self._uncompressed_size.pack(self.uncompressed_size),
            self._compressed_size.pack(self.compressed_size),
        ]))

    # noinspection PyArgumentList
    def pack(self) -> bytes:
        return b''.join((
            self._fn_size.pack(self.fn_size),
            self._fn_encoded,
            self._uncompressed_size.pack(self.uncompressed_size),
            self._compressed_size.pack(self.compressed_size),
            self.offset.to_bytes(1),
            self.pack_encoding()
        ))

    # noinspection PyArgumentList
    @classmethod
    def unpack(cls, stream: typing.BinaryIO) -> 'Header':
        return cls(
            stream.read(cls._fn_size.unpack(stream)),
            cls._compressed_size.unpack(stream),
            cls._uncompressed_size.unpack(stream),
            int.from_bytes(stream.read(1)),
            cls.unpack_encoding(stream)
        )

    def pack_encoding(self) -> bytes:
        encoding = b''

        for byte, bits in self.encoding.items():
            encoding += byte + bytes(bits) + b'\n'

        return encoding + b'\n'

    @staticmethod
    def unpack_encoding(stream: typing.BinaryIO) -> dict:
        encoding, i = {}, 0

        while i < 4096:
            byte = stream.read(1)

            if byte == b'\n':
                return encoding

            encoding[byte] = BitArray()
            while (bit := stream.read(1)) != b'\n' and i < 4096:
                encoding[byte] += bin(bit[0])
                i += 1

            i += 1

        raise ValueError("couldn't find encoding end")


class ZipInfo:
    def __init__(self):
        pass


class Zip:
    def __init__(self, zip_fp: str, files: list[str]):
        self.zip_fp = zip_fp
        self.files = list(map(os.path.abspath, files))

    # noinspection PyArgumentList
    def create(self):
        z = open(self.zip_fp, 'wb')

        for file in self.files:
            encoder = compression.Encoder(ConstBitStream(filename=file))
            header = Header(
                os.path.basename(file),
                os.stat(file).st_size,
                encoder.encoded_size,
                encoder.encoded_offset,
                encoder.encoding
            )
            z.write(header.pack())

            for payload in encoder.encode():
                z.write(payload)

        z.close()
