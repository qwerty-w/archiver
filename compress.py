import time

from bitstring import ConstBitStream, BitArray as _BitArray
import typing


class BitArray(_BitArray):
    def __hash__(self):
        return hash(self.bin)


class InnerNode:  # PlaceHolder
    def __str__(self):
        return '<InnerNode>'


class Node:
    def __init__(self, value: bytes | InnerNode, freq: int, *, left: 'Node' = None, right: 'Node' = None):
        self.left = left  # 0
        self.right = right  # 1
        self.value = value
        self.freq = freq

    def __str__(self):
        return str(self.value)


class Wrapper:
    LEFT = 0
    RIGHT = 1

    def __init__(self, node: Node):
        self.node = node
        self.state = self.LEFT

    def next_left(self):
        return self.state == self.LEFT

    def next_right(self):
        return self.state == self.RIGHT

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f'{self.node}:{self.state}'


class _DecodeIterator:
    def __init__(self, encoded: 'Decoder', segment_size: int):
        self.encoded = encoded
        self.segment_size = segment_size

        self.table = dict(zip(self.encoded.encoding.values(), self.encoded.encoding.keys()))
        self.lb = self.encoded.stream.pos + self.encoded.bit_length
        self.code = BitArray()

    def __iter__(self):
        return self

    def __next__(self) -> bytes:
        if self.encoded.stream.pos + self.segment_size > self.lb:  # todo: check
            size = self.lb - self.encoded.stream.pos

        else:
            size = self.segment_size

        if size <= 0:
            raise StopIteration

        rv = b''
        for bit in self.encoded.stream.read(f'bin{size}'):
            self.code += '0b' + bit

            if self.code in self.table:
                rv += self.table[self.code]
                self.code.clear()

        return rv


class Decoder:
    def __init__(self, encoding: dict[bytes, BitArray], stream: ConstBitStream, bit_length: int):
        self.encoding = encoding
        self.stream = stream
        self.bit_length = bit_length
        self.segment_size = 64  # bit

    def __iter__(self):
        return self.decode()

    def decode(self, segment_size: int = 64) -> typing.Iterator:
        return _DecodeIterator(self, segment_size)


class Encoder:  # HuffmanCoding
    def __init__(self, stream: ConstBitStream):
        self.stream = stream
        self.root = self.build_tree()
        self.encoding = self.get_encoding()
        self._encoded_offset = None

    @staticmethod
    def _pop2smallest(nodes: list[Node]) -> tuple[Node, Node]:
        if len(nodes) < 2:
            raise ValueError('nodes.length < 2')

        i1, i2 = range(2)
        for i in range(len(nodes)):
            n = nodes[i]

            if n.freq < nodes[i1].freq:
                i1, i2 = i, i1

            elif n.freq < nodes[i2].freq:
                i2 = i

        return nodes.pop(i1), nodes.pop(i2 - 1 if i2 > i1 else i2)

    def build_tree(self) -> Node:
        freq = {}

        while self.stream.pos < self.stream.length:
            byte = self.stream.read('bytes1')

            if byte in freq:
                freq[byte].freq += 1

            else:
                freq[byte] = Node(byte, 1)
        self.stream.pos = 0

        nodes = list(freq.values())
        while len(nodes) > 1:
            left, right = self._pop2smallest(nodes)
            nodes.append(Node(InnerNode(), left.freq + right.freq, left=left, right=right))

        root = nodes.pop()
        return root

    def get_encoding(self) -> dict[bytes, BitArray]:
        encoding = {}

        stack = [Wrapper(self.root)]
        code = BitArray()
        while stack:
            wrapper = stack[-1]
            current = wrapper.node

            if current.left and wrapper.next_left():
                wrapper.state = wrapper.RIGHT
                stack.append(Wrapper(current.left))
                code += '0b0'

            elif current.right and wrapper.next_right():
                wrapper.state = None
                stack.append(Wrapper(current.right))
                code += '0b1'

            else:
                n = stack.pop().node

                if not isinstance(n.value, InnerNode):
                    encoding[n.value] = code

                code = code[:-1]

        return encoding

    def encode(self) -> bytes:  # todo: refactor me please
        rv = BitArray()

        while self.stream.pos < self.stream.length:
            byte = self.stream.read('bytes1')
            rv += self.encoding[byte]

            if rv.length >= 8:
                b, rv = rv[:8].tobytes(), rv[8:]
                yield b

        self._encoded_offset = 8 - rv.length
        yield rv.tobytes()

    def get_encoded_offset(self):
        if self._encoded_offset:
            return self._encoded_offset

        raise RuntimeError('encoded offset only can be received after .encoded')


def time_wrap(func, *args, _round: int = 2):
    t0 = time.time()
    rv = func(*args)
    return int((time.time() - t0) * 10 ** _round) / 10 ** _round, rv


def filetest():
    # stream = ConstBitStream(filename='test')
    # encoder = Encoder(stream)
    #
    # with open('test_compressed', 'wb') as f:
    #     for b in encoder.encode():
    #         b: bytes
    #         f.write(b)
    #
    # print(sorted(encoder.encoding.items(), key=lambda x: x[1].length))
    # print(encoder.get_encoded_offset())

    encoding = dict([(b't', BitArray('0b000')), (b' ', BitArray('0b110')), (b'y', BitArray('0x4')), (b'e', BitArray('0x5')),
     (b'r', BitArray('0x7')), (b'a', BitArray('0x8')), (b'h', BitArray('0xb')), (b'w', BitArray('0b00110')),
     (b'o', BitArray('0b00111')), (b'd', BitArray('0b01101')), (b'\r', BitArray('0b10011')),
     (b'\n', BitArray('0b10100')), (b'i', BitArray('0b10101')), (b's', BitArray('0b11101')),
     (b'n', BitArray('0b11110')), (b',', BitArray('0b001001')), (b'u', BitArray('0b001010')),
     (b'l', BitArray('0b011000')), (b'g', BitArray('0b100100')), (b'?', BitArray('0b100101')),
     (b'f', BitArray('0b111001')), (b'm', BitArray('0b111110')), (b'p', BitArray('0b0010111')),
     (b'W', BitArray('0b1110000')), (b'b', BitArray('0b1111110')), (b'C', BitArray('0x20')), (b'&', BitArray('0x21')),
     (b'v', BitArray('0x22')), (b':', BitArray('0x23')), (b'k', BitArray('0x65')), (b'.', BitArray('0x66')),
     (b'A', BitArray('0x67')), (b'c', BitArray('0xe2')), (b'T', BitArray('0xe3')), (b'I', BitArray('0xfe')),
     (b'D', BitArray('0xff')), (b';', BitArray('0b001011000')), (b'B', BitArray('0b001011001')),
     (b'O', BitArray('0b001011010')), (b'z', BitArray('0b001011011')), (b"'", BitArray('0b011001000')),
     (b'L', BitArray('0b011001001'))])
    offset = 3

    stream = ConstBitStream(filename='test_compressed')
    decoder = Decoder(encoding, stream, stream.length - offset)

    with open('original_test', 'wb') as f:
        for b in decoder.decode():
            f.write(b)

    import hashlib
    with open('test', 'rb') as f:
        h1 = hashlib.sha256(f.read()).hexdigest()

    with open('original_test', 'rb') as f:
        h2 = hashlib.sha256(f.read()).hexdigest()

    print('\n'.join([h1, h2]))
    print(h1 == h2)
    assert h1 == h2


def main():
    string = b'Mama mila ramu'
    print(f'String [{len(string)}]:', string, end='\n\n')

    e = Encoder(ConstBitStream(string))

    encoded = b''
    for i in e.encode():
        encoded += i
    offset = e.get_encoded_offset()

    # print(encoded)

    e = Decoder(e.encoding, ConstBitStream(encoded), len(encoded) * 8 - offset)

    decoded = b''
    for i in e.decode():
        decoded += i

    print(decoded)

    # t_e, encoded = time_wrap(h.encode)
    # t_d, decoded = time_wrap(HuffmanCoding.decode, h.encoding, encoded)
    #
    # print('Encoding:', dict(map(lambda x: (chr(x[0]), x[1]), h.encoding.items())))
    # print(f'Encoded string [{t_e}/{len(encoded)} bytes]:', bin(int.from_bytes(encoded, 'big', signed=False)))
    # print(f'Decoded string [{t_d}/{len(string)} bytes]:', decoded)


if __name__ == '__main__':
    filetest()
