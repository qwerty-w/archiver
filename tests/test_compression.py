import os
import pytest
import hashlib
from huffman import compression


TEST_FILE = './test'


def test_encode():
    pass


def test_decode():
    pass


def test_file_compression():
    compressed_filename = TEST_FILE + '_compressed'
    uncompressed_filename = TEST_FILE + '_uncompressed'
    with open(compressed_filename, 'wb') as f:
        stream = compression.ConstBitStream(filename=TEST_FILE)
        encoder = compression.Encoder(stream)
        for byte in encoder.encode():
            f.write(byte)

    with open(uncompressed_filename, 'wb') as f:
        stream = compression.ConstBitStream(filename=compressed_filename,
                                            length=encoder.encoded_size * 8 - encoder.encoded_offset)
        decoder = compression.Decoder(encoder.encoding, stream)
        for _bytes in decoder.decode():
            f.write(_bytes)

    hs = []
    for fn in [uncompressed_filename, TEST_FILE]:
        h = hashlib.sha256()

        with open(fn, 'rb') as f:
            while x := f.read(1024):
                h.update(x)

        hs.append(h)

    assert hs[0].hexdigest() == hs[1].hexdigest()
