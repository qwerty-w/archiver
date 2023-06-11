import time


class InnerNode:  # PlaceHolder
    def __str__(self):
        return '<InnerNode>'


class Node:
    def __init__(self, value: str | InnerNode, freq: int, *, left: 'Node' = None, right: 'Node' = None):
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


class Huffman:
    def __init__(self, string: str):
        self.string = string
        self.root = self.build_tree()
        self.encoding = self.get_encoding()

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
        for character in self.string:
            if character in freq:
                freq[character].freq += 1

            else:
                freq[character] = Node(character, 1)

        nodes = list(freq.values())
        while len(nodes) > 1:
            left, right = self._pop2smallest(nodes)
            nodes.append(Node(InnerNode(), left.freq + right.freq, left=left, right=right))

        root = nodes.pop()
        return root

    def get_encoding(self) -> dict[str, str]:
        encoding = {}

        stack: list[Wrapper] = [Wrapper(self.root)]
        code = ''
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

                if not isinstance(n.value, InnerNode):
                    encoding[n.value] = code

                code = code[:-1]

        return encoding

    def encode(self) -> str:
        return ''.join(self.encoding[character] for character in self.string)

    @staticmethod
    def decode(encoding: dict[str, str], encoded_string: str):
        reversed_encoding = dict(zip(encoding.values(), encoding.keys()))

        rv = code = ''
        for bit in encoded_string:
            code += bit

            if code in reversed_encoding:
                rv += reversed_encoding[code]
                code = ''

        return rv


def time_wrap(func, *args, _round: int = 2):
    t0 = time.time()
    rv = func(*args)
    return int((time.time() - t0) * 10 ** _round) / 10 ** _round, rv


def main():
    string = 'Mama mila ramu'
    print('String:', string)

    h = Huffman(string)
    t_e, encoded = time_wrap(h.encode)
    t_d, decoded = time_wrap(Huffman.decode, h.encoding, encoded)

    print('Encoding:', h.encoding)
    print(f'Encoded string [{t_e}/{len(encoded) / 8} bytes]:', encoded)
    print(f'Decoded string [{t_d}/{len(string)} bytes]:', decoded)


if __name__ == '__main__':
    main()
