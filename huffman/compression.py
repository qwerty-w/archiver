import typing
from abc import ABC
from functools import partial
from bitstring import BitStream, ConstBitStream


class InnerNode:  # PlaceHolder
    def __str__(self):
        return '<InnerNode>'


class NYT:
    pass


class Node:
    def __init__(self, value: bytes | type[InnerNode, NYT], freq: int, *,
                 parent: 'Node' = None, left: 'Node' = None, right: 'Node' = None):

        self.value = value
        self.freq = freq
        self.parent = parent
        self.left = left  # 0
        self.right = right  # 1

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
    <FULL encoding size:2> <max bits length:1> <0111 1001001> <...>
                                                size  value
    """
    def __init__(self, table: dict[bytes, str], max_code_length: int = None):
        self.table = table
        self.max_code_length = max_code_length if max_code_length is not None else self.get_max_code_length()

    def __eq__(self, other):
        return self.table == other.table if isinstance(other, Encoding) else self.table == other

    def get_max_code_length(self) -> int:
        return max(map(len, self.table.values()))

    def get_reversed_table(self) -> dict[str, bytes]:
        return dict(zip(self.table.values(), self.table.keys()))

    def pack(self) -> bytes:
        code_length_size = (self.max_code_length - 1).bit_length()

        if code_length_size >= 256:
            raise ValueError('max code length is too big')

        auto = ''
        for byte, bits in self.table.items():
            code_length = len(bits) - 1

            if code_length < 0:
                raise ValueError(f'code for byte {byte} is empty')

            auto += bin(byte[0])[2:].zfill(8) + bin(code_length)[2:].zfill(code_length_size) + bits

        auto = auto + '0' * (8 - len(auto) % 8 if len(auto) % 8 else 0)
        rv = b''.join(int(auto[i:i + 8], 2).to_bytes(1) for i in range(0, len(auto), 8))

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
            table[byte] = data.read(f'bits{code_length}').bin

        return cls(table)

    def __reversed__(self):
        return self.get_reversed_table()


class TwoPassEncoder:  # HuffmanCoding
    def __init__(self, stream: typing.BinaryIO):
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
        freq, pos = {}, self.stream.tell()
        while byte := self.stream.read(1):
            if byte in freq:
                freq[byte].freq += 1

            else:
                freq[byte] = Node(byte, 1)
        self.stream.seek(pos)

        nodes = list(freq.values())
        while len(nodes) > 1:
            left, right = self._pop2smallest(nodes)
            nodes.append(Node(InnerNode, left.freq + right.freq, left=left, right=right))

        root = nodes.pop()
        return root

    def _get_encs(self) -> tuple[Encoding, int, int]:
        encoding = {}

        stack, code = [Wrapper(self.root)], ''
        max_code_length = bit_size = 0
        while stack:
            wrapper = stack[-1]
            current = wrapper.node

            if current.left and wrapper.next_left():
                wrapper.state = wrapper.RIGHT
                stack.append(Wrapper(current.left))
                code += '0'

            elif current.right and wrapper.next_right():
                wrapper.state = None
                stack.append(Wrapper(current.right))
                code += '1'

            else:
                n = stack.pop().node

                if n.value is not InnerNode:
                    encoding[n.value] = code
                    bit_size += len(code) * n.freq
                    if len(code) > max_code_length:
                        max_code_length = len(code)

                code = code[:-1]

        remainder = bit_size % 8
        return (Encoding(encoding, max_code_length),
                bit_size // 8 + (remainder > 0),
                8 - remainder if remainder else 0)

    # noinspection PyArgumentList
    def encode(self) -> typing.Generator[bytes, None, None]:
        yv = ''

        while byte := self.stream.read(1):
            if len(yv) >= 8:
                b, yv = int(yv[:8], 2).to_bytes(1), yv[8:]
                yield b

            yv += self.encoding.table[byte]

        yv += '0' * (8 - len(yv) % 8 if len(yv) % 8 else 0)
        yield int(yv, 2).to_bytes(len(yv) // 8)


class TwoPassDecoder:
    def __init__(self, encoding: Encoding, stream: ConstBitStream, segment_size: int = 64):
        self.encoding = encoding
        self.stream = stream
        self.segment_size = segment_size
        self.table = self.encoding.get_reversed_table()

    def __iter__(self):
        return next(self)

    def __next__(self) -> typing.Generator[bytes, None, None]:
        code, write_segment = '', b''
        for bit in self.stream:
            code += '1' if bit else '0'
            if code in self.table:
                write_segment += self.table[code]
                code = ''

                if len(write_segment) >= self.segment_size:
                    yield write_segment
                    write_segment = b''

        yield write_segment

    def decode(self):
        return iter(self)


class OnePassMethod(ABC):
    def __init__(self):
        self.root = self.nyt = Node(NYT, 0)
        self.nodes = []
        self.seen: list[Node | None] = [None] * 256

        self.segment_size = 64  # bytes

    def find_largest_node(self, freq: int) -> Node:
        for n in reversed(self.nodes):
            if n.freq == freq:
                return n

    def swap_node(self, n1: Node, n2: Node):
        i1, i2 = self.nodes.index(n1), self.nodes.index(n2)
        self.nodes[i1], self.nodes[i2] = self.nodes[i2], self.nodes[i1]
        n1.parent, n2.parent = n2.parent, n1.parent

        # replace parent's child
        for _n, _is, _rp in [
            (n1, n2, n1), (n2, n1, n2)
        ]:
            setattr(_n.parent, 'left' if _n.parent.left is _is else 'right', _rp)

    def insert(self, value: bytes):
        node = self.seen[value[0]]

        if node is None:
            n = Node(value, 1)
            inner = Node(InnerNode, 1, parent=self.nyt.parent, left=self.nyt, right=n)
            n.parent = self.nyt.parent = inner

            if inner.parent:
                inner.parent.left = inner

            else:
                self.root = inner

            for _n in [inner, n]:
                self.nodes.insert(0, _n)

            self.seen[value[0]] = n
            node = inner.parent

        while node is not None:
            largest = self.find_largest_node(node.freq)

            if node not in [largest, largest.parent] and largest is not node.parent:
                self.swap_node(node, largest)

            node.freq = node.freq + 1
            node = node.parent


class OnePassEncoder(OnePassMethod):
    def __init__(self, stream: typing.BinaryIO):
        super().__init__()
        self.stream = stream
        self._size = self._offset = None

    def _dfs_find(self, v: bytes | type[NYT], node: Node, code: str = '') -> str:
        if node.left is None and node.right is None:
            return code if node.value == v else ''

        _code = ''
        if node.left:
            _code = self._dfs_find(v, node.left, code + '0')
        if not _code and node.right:
            _code = self._dfs_find(v, node.right, code + '1')

        return _code

    def get_code(self, v: bytes | type[NYT]):
        return self._dfs_find(v, self.root, '')

    def get_size_inf(self) -> tuple[int, int]:
        if self._size is None or self._offset is None:
            raise RuntimeError('size and offset can only be obtained after end of .encode()')

        return self._size, self._offset

    # noinspection PyArgumentList
    def encode(self) -> typing.Generator[bytes, None, None]:
        code = ''
        size = 0

        while data := self.stream.read(self.segment_size):
            for integer in data:
                byte = integer.to_bytes(1)

                code += self.get_code(byte) if self.seen[integer] \
                    else self.get_code(NYT) + bin(integer)[2:].zfill(8)
                self.insert(byte)

                if len(code) > 8:
                    yv, code = code[:8], code[8:]
                    size += 1
                    yield int(yv, 2).to_bytes(1)

        self._offset = 8 - len(code) % 8 if len(code) % 8 else 0
        code += '0' * self._offset
        yv = int(code, 2).to_bytes(len(code) // 8)
        self._size = size + len(yv)
        yield yv


class OnePassDecoder(OnePassMethod):
    def __init__(self, stream: ConstBitStream):
        super().__init__()
        self.stream = stream

    def decode(self) -> typing.Generator[bytes, None, None]:
        # noinspection PyRedundantParentheses
        yield (fb := self.stream.read('bytes1'))
        self.insert(fb)
        node = self.root

        while self.stream.pos < self.stream.length:
            node = node.right if self.stream.read('bool') else node.left
            value = node.value

            if value is InnerNode:
                continue

            if value is NYT:
                value = self.stream.read('bytes1')

            yield value
            self.insert(value)
            node = self.root
