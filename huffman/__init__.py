from bitstring import ConstBitStream, BitArray as _BitArray


class BitArray(_BitArray):
    def __hash__(self):
        return hash(self.bin)

