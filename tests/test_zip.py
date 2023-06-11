import random
import pytest
from io import BytesIO
from huffman import zip


@pytest.fixture(params=range(1, 8))
def varint(request):
    return zip.VarInt(request.param)


@pytest.fixture(params=[lambda vi: vi.max_value, lambda vi: 256 ** vi.size])
def rand_varint(request, varint):
    return varint, random.randint(0, request.param(varint))


def test_varint(rand_varint):
    varint, num = rand_varint
    stream = BytesIO(varint.pack(num))
    assert varint.unpack(stream) == num


@pytest.fixture
def h(request):
    return zip.Header(*TestHeader.test_args)


class TestHeader:
    test_args = [
        'filename', 5729813, 279813, 4, {
            b'\xf0': zip.BitArray('0b101010'),
            b'\xee': zip.BitArray('0b1010'),
            b'\x24': zip.BitArray('0b101'),
            b'\x00': zip.BitArray('0b1010101'),
            b'\xff': zip.BitArray('0b001')
        }
    ]
    test_serialized_header = b'\x08filename\xfc\x57\x6e\x15\xfc\x04\x45\x05\x04\xf0\x01\x00\x01\x00\x01\x00' \
                             b'\n\xee\x01\x00\x01\x00\n\x24\x01\x00\x01\n\x00\x01\x00\x01\x00\x01\x00\x01\n' \
                             b'\xff\x00\x00\x01\n\n'

    def test_offset_pos(self, h):
        assert h.get_offset_position() == 17

    def test_encoding_pack_unpack(self, h):
        assert h.pack_encoding() == self.test_serialized_header[18:]
        assert h.unpack_encoding(BytesIO(self.test_serialized_header[18:])) == self.test_args[-1]

    def test_pack_unpack(self, h):
        assert h.pack() == self.test_serialized_header

        unpacked = h.unpack(BytesIO(self.test_serialized_header))
        for i in filter(lambda x: not x.startswith('__') and not callable(getattr(h, x)), dir(h)):
            assert getattr(h, i) == getattr(unpacked, i)
