#!/usr/bin/env python

import math

def gcd(a,b):
    while b:
        a,b = b, a % b
    return a

def lcm(a,b):
    return a * b / gcd(a, b)

def _npadding_bytes(pkt_byte_len, samples_per_symbol, bits_per_symbol):
    """
    Generate sufficient padding such that each packet ultimately ends
    up being a multiple of 512 bytes when sent across the USB.  We
    send 4-byte samples across the USB (16-bit I and 16-bit Q), thus
    we want to pad so that after modulation the resulting packet
    is a multiple of 128 samples.

    @param ptk_byte_len: len in bytes of packet, not including padding.
    @param samples_per_symbol: samples per bit (1 bit / symbolwidth GMSK)
    @type samples_per_symbol: int
    @param bits_per_symbol: bits per symbol (log2(modulation order))
    @type bits_per_symbol: int

    @returns number of bytes of padding to append.
    """
    modulus = 128
    byte_modulus = lcm(modulus/8, samples_per_symbol) * bits_per_symbol / samples_per_symbol
    r = pkt_byte_len % byte_modulus
    if r == 0:
        return 0
    return byte_modulus - r

if __name__ == '__main__':
    for i in range(512):
        print i, _npadding_bytes(512 + i, 2, 1)
