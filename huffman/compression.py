import time
import typing
from functools import partial
from huffman import BitArray, BitStream, ConstBitStream


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

    def next_left(self) -> bool:
        return self.state == self.LEFT

    def next_right(self) -> bool:
        return self.state == self.RIGHT

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f'{self.node}:{self.state}'


# noinspection PyArgumentList
class Encoding:
    """
    <>
    """
    def __init__(self, table: dict[bytes, BitArray], max_code_length: int = None):
        self.table = table
        self.max_code_length = max_code_length if max_code_length is not None else self.get_max_code_length()

    def get_max_code_length(self) -> int:
        return sum(map(len, self.table.values()))

    def get_reversed_table(self) -> dict[BitArray, bytes]:
        return dict(zip(self.table.values(), self.table.keys()))

    # <FULL encoding size:2> <max bits length:1(7 for example)> <0111 1001001>
    #                                                            size  value
    def pack(self) -> bytes:
        code_length_size = (self.max_code_length - 1).bit_length()

        if code_length_size >= 256:
            raise ValueError('max code length is too big')

        barray = BitArray()
        for byte, bits in self.table.items():
            code_length = len(bits) - 1

            if code_length < 0:
                raise ValueError(f'code for byte {byte} is empty')

            for val in [
                '0x' + byte.hex(),
                '0b' + '0' * (code_length_size - code_length.bit_length()) + bin(code_length)[2:],
                bits
            ]: barray.append(val)

        rv = barray.tobytes()

        if len(rv) >= 256 ** 2:
            raise ValueError('encoding is too big')

        return b''.join((
            len(rv).to_bytes(2),
            code_length_size.to_bytes(1),
            rv
        ))

    @staticmethod
    def _fill_bitstream(to_read: typing.BinaryIO, to_write: BitStream, end: int, required_bit_size: int):
        while len(to_write) - to_write.pos < required_bit_size and to_read.tell() < end and (read := to_read.read(1)):
            to_write += read

    @classmethod
    def unpack(cls, stream: typing.BinaryIO) -> 'Encoding':
        encoding_size = int.from_bytes(stream.read(2))
        code_length_size = int.from_bytes(stream.read(1))

        data, table = BitStream(), {}
        index, end = 0, stream.tell() + encoding_size
        fill = partial(cls._fill_bitstream, stream, data, end)
        while stream.tell() < end:
            fill(8)
            byte = data.read('bytes1')
            fill(code_length_size)
            code_length = data.read(f'uint{code_length_size}') + 1
            fill(code_length)
            table[byte] = BitArray(data.read(f'bits{code_length}'))

        return cls(table)

    def __reversed__(self):
        return self.get_reversed_table()


class Encoder:  # HuffmanCoding
    def __init__(self, stream: ConstBitStream):
        self.stream = stream
        self.root = self.build_tree()
        self.encoding, self.encoded_size, self.encoded_offset = self._get_encs()

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

    def _get_encs(self) -> tuple[Encoding, int, int]:
        encoding = {}

        stack = [Wrapper(self.root)]
        code = BitArray()
        max_code_length = bit_size = 0
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
                    bit_size += len(code) * n.freq
                    if len(code) > max_code_length:
                        max_code_length = len(code)

                code = code[:-1]

        remainder = bit_size % 8
        return (Encoding(encoding, max_code_length),
                bit_size // 8 + (remainder > 0),
                8 - remainder if remainder else 0)

    def encode(self) -> typing.Generator[bytes, None, None]:
        yv = BitArray()

        while self.stream.pos < self.stream.length:
            if yv.length >= 8:
                b, yv = yv[:8].tobytes(), yv[8:]
                yield b

            byte = self.stream.read('bytes1')
            yv += self.encoding.table[byte]

        yield yv.tobytes()


class Decoder:
    def __init__(self, encoding: Encoding, stream: ConstBitStream, segment_size: int = 64):
        self.encoding = encoding
        self.stream = stream
        self.segment_size = segment_size
        self.table = self.encoding.get_reversed_table()

    def __iter__(self):
        return next(self)

    def __next__(self) -> typing.Generator[bytes, None, None]:
        code = BitArray()

        for segment in self.stream.cut(self.segment_size):
            for bit in segment.bin:
                code += '0b' + bit

                if code in self.table:
                    yield self.table[code]
                    code.clear()

    def decode(self):
        return iter(self)


def time_wrap(func, *args, _round: int = 2):
    t0 = time.time()
    rv = func(*args)
    return int((time.time() - t0) * 10 ** _round) / 10 ** _round, rv


def main():
    # enc = Encoding({
    #     b'\xf0': BitArray('0b10101011'),
    #     b'\xee': BitArray('0b1010'),
    #     b'\x24': BitArray('0b101'),
    #     b'\x00': BitArray('0b101010'),
    #     b'\xff': BitArray('0b001')
    # })
    # e = enc.pack()
    # BitArray(e)[4 * 8:].pp()
    # from io import BytesIO
    # d = enc.unpack(BytesIO(e))
    # print(d.table)
    # return

    string = b'Mama mila ramu'
    print(f'String [{len(string)}]:', string, end='\n\n')

    e = Encoder(ConstBitStream(string))

    encoded = b''
    for i in e.encode():
        encoded += i

    # print(encoded)

    from io import BytesIO
    stream: ConstBitStream = ConstBitStream(BytesIO(encoded), length=e.encoded_size * 8 - e.encoded_offset)
    d = Decoder(e.encoding, stream)

    decoded = b''
    for i in d.decode():
        decoded += i

    print(decoded)

    # t_e, encoded = time_wrap(h.encode)
    # t_d, decoded = time_wrap(HuffmanCoding.decode, h.encoding, encoded)
    #
    # print('Encoding:', dict(map(lambda x: (chr(x[0]), x[1]), h.encoding.items())))
    # print(f'Encoded string [{t_e}/{len(encoded)} bytes]:', bin(int.from_bytes(encoded, 'big', signed=False)))
    # print(f'Decoded string [{t_d}/{len(string)} bytes]:', decoded)


if __name__ == '__main__':
    main()
