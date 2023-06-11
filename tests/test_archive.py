import random
import pytest
from io import BytesIO
from huffman import archive, compression


@pytest.fixture(params=range(1, 8))
def varint(request):
    return archive.VarInt(request.param)


@pytest.fixture(params=[lambda vi: vi.max_value, lambda vi: 256 ** vi.size])
def rand_varint(request, varint):
    return varint, random.randint(0, request.param(varint))


def test_varint(rand_varint):
    varint, num = rand_varint
    stream = BytesIO(varint.pack(num))
    assert varint.unpack(stream) == num


@pytest.fixture
def h(request):
    return archive.Header(*TestHeader.test_args)


class TestHeader:
    test_args = [
        'filename', 5729813, 279813, 4, 'twopass', compression.Encoding({
            b'\xf0': '101010',
            b'\xee': '1010',
            b'\x24': '101',
            b'\x00': '1010101',
            b'\xff': '001'
        })
    ]
    test_serialized_header = b'\x08filename\xfc\x57\x6e\x15\xfc\x04\x45\x05\x04\x01\x00\x0a\x03\xf0\xb5\x77' \
                             b'\x3a\x24\x54\x03\x55\xff\x44'

    def test_compressed_size_pos(self, h):
        assert h.get_compressed_size_pos() == 13

    def test_encoding_pack_unpack(self, h):
        assert self.test_serialized_header[19:] == h.encoding.pack()
        assert compression.Encoding.unpack(BytesIO(self.test_serialized_header[19:])).table == self.test_args[-1].table

    def test_pack_unpack(self, h):
        assert h.pack() == self.test_serialized_header

        unpacked = h.unpack(BytesIO(self.test_serialized_header))
        for i in filter(lambda x: not x.startswith('__') and not callable(getattr(h, x)), dir(h)):
            assert getattr(h, i) == getattr(unpacked, i)
