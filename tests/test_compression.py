import io
import os
import hashlib
from dataclasses import dataclass

import pytest
from huffman import compression


@dataclass
class File:
    path: str
    original_hash: str
    encoded_hash: dict[str, str]


class TestCompression:
    TABLE = {
        b'\xff': '101010',
        b'\xef': '1010',
        b'\x12': '101',
        b'\x00': '1010101',
        b'\xf0': '001',
        b'\x04': '100'
    }
    REVERSED_TABLE = {
        '101010': b'\xff',
        '1010': b'\xef',
        '101': b'\x12',
        '1010101': b'\x00',
        '001': b'\xf0',
        '100': b'\x04'
    }
    MAX_CODE_LENGTH = 7
    PACKED_ENCODING = b'\x00\x0c\x03\xff\xb5\x77\xba\x12\x54\x03\x55\xf0\x44\x11\x40'

    def test_encoding(self):
        enc = compression.Encoding(self.TABLE)
        assert enc.max_code_length == self.MAX_CODE_LENGTH == \
               compression.Encoding(self.TABLE, self.MAX_CODE_LENGTH).max_code_length
        assert enc.pack() == self.PACKED_ENCODING
        assert enc.get_reversed_table() == self.REVERSED_TABLE == reversed(enc)

        unpacked = compression.Encoding.unpack(io.BytesIO(self.PACKED_ENCODING))
        assert unpacked.table == self.TABLE
        assert unpacked.max_code_length == self.MAX_CODE_LENGTH
        assert unpacked == enc  # __eq__ test

    FILES = [File('data/Tyger',
                  'D2870D04898C66CD04EFA28E3D8A716889B9024791DE76F47819835140479D94',
                  {'onepass': '6E7930E2933E0721C171CE2FFEA1A02B749D3C4037565D282B9A38CA372F6B3B',
                   'twopass': '7F7DB87C70BBDF8A7F5E8D8BE6940E6D03FCCA6458B500C89EF1ABBBBCE5C39D'}),
             File('data/Yesenin.txt',
                  '0E256558C951D22A3A15DC4739C810A10D461D26ABAF44463043E227421AE03F',
                  {'onepass': '5E36072D7BE7473EFC1027C0D186C91DB4853FFB748201B959ECF8FDC44904AB',
                   'twopass': 'B526861064F417EAC94ADD3CC4A5F6E95732A70665ABFFCAF68CD41BA96BB2DC'})]

    @pytest.fixture(params=FILES, ids=lambda x: f'"{x.path}"')
    def file(self, request):
        return request.param

    @pytest.fixture(params=['onepass', 'twopass'])
    def method(self, request):
        return request.param

    # noinspection PyUnboundLocalVariable
    def test_encode_decode(self, file: File, method: str):
        with open(file.path, 'rb') as f:
            encoder = {
                'onepass': compression.OnePassEncoder,
                'twopass': compression.TwoPassEncoder
            }[method](f)

            compressed_fp = file.path + '.compressed'
            with open(compressed_fp, 'wb') as wf:
                hsh = hashlib.sha256()
                for segment in encoder.encode():
                    hsh.update(segment)
                    wf.write(segment)

            assert hsh.hexdigest().upper() == file.encoded_hash[method]

        match method:
            case 'onepass':
                size, offset = encoder.get_size_inf()

            case 'twopass':
                size, offset = encoder.encoded_size, encoder.encoded_offset

        decoder_stream = compression.ConstBitStream(filename=compressed_fp, length=size * 8 - offset)

        match method:
            case 'onepass':
                decoder = compression.OnePassDecoder(decoder_stream)

            case 'twopass':
                enc = compression.Encoding(encoder.encoding.table)
                decoder = compression.TwoPassDecoder(enc, decoder_stream)

        hsh = hashlib.sha256()
        for segment in decoder.decode():
            hsh.update(segment)

        assert hsh.hexdigest().upper() == file.original_hash
        decoder_stream._clear()
        os.remove(compressed_fp)
