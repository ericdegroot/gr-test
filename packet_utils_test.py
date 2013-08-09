import struct
from gnuradio import digital

def conv_packed_binary_string_to_1_0_string(s):
    """
    '\xAF' --> '10101111'
    """
    r = []
    for ch in s:
        x = ord(ch)
        for i in range(7,-1,-1):
            t = (x >> i) & 0x1
            r.append(t)

    return ''.join(map(lambda x: chr(x + ord('0')), r))

def is_1_0_string(s):
    if not isinstance(s, str):
        return False
    for ch in s:
        if not ch in ('0', '1'):
            return False
    return True

def conv_1_0_string_to_packed_binary_string(s):
    """
    '10101111' -> ('\xAF', False)

    Basically the inverse of conv_packed_binary_string_to_1_0_string,
    but also returns a flag indicating if we had to pad with leading zeros
    to get to a multiple of 8.
    """
    if not is_1_0_string(s):
        raise ValueError, "Input must be a string containing only 0's and 1's"
    
    # pad to multiple of 8
    padded = False
    rem = len(s) % 8
    if rem != 0:
        npad = 8 - rem
        s = '0' * npad + s
        padded = True

    assert len(s) % 8 == 0

    r = []
    i = 0
    while i < len(s):
        t = 0
        for j in range(8):
            t = (t << 1) | (ord(s[i + j]) - ord('0'))
        r.append(chr(t))
        i += 8
    return (''.join(r), padded)

default_access_code = \
  conv_packed_binary_string_to_1_0_string('\xAC\xDD\xA4\xE2\xF2\x8C\x20\xFC')
preamble = \
  conv_packed_binary_string_to_1_0_string('\xA4\xF2')

def make_header(payload_len, whitener_offset=0):
    # Upper nibble is offset, lower 12 bits is len
    val = ((whitener_offset & 0xf) << 12) | (payload_len & 0x0fff)
    #print "offset =", whitener_offset, " len =", payload_len, " val=", val
    return struct.pack('!HH', val, val)

def make_packet(payload, samples_per_symbol, bits_per_symbol,
                access_code=default_access_code, pad_for_usrp=True,
                whitener_offset=0, whitening=True):
    """
    Build a packet, given access code, payload, and whitener offset

    @param payload:               packet payload, len [0, 4096]
    @param samples_per_symbol:    samples per symbol (needed for padding calculation)
    @type  samples_per_symbol:    int
    @param bits_per_symbol:       (needed for padding calculation)
    @type bits_per_symbol:        int
    @param access_code:           string of ascii 0's and 1's
    @param whitener_offset        offset into whitener string to use [0-16)
    
    Packet will have access code at the beginning, followed by length, payload
    and finally CRC-32.
    """
    if not is_1_0_string(access_code):
        raise ValueError, "access_code must be a string containing only 0's and 1's (%r)" % (access_code,)

    if not whitener_offset >=0 and whitener_offset < 16:
        raise ValueError, "whitener_offset must be between 0 and 15, inclusive (%i)" % (whitener_offset,)

    (packed_access_code, padded) = conv_1_0_string_to_packed_binary_string(access_code)
    (packed_preamble, ignore) = conv_1_0_string_to_packed_binary_string(preamble)
    
    payload_with_crc = crc.gen_and_append_crc32(payload)
    #print "outbound crc =", string_to_hex_list(payload_with_crc[-4:])

    L = len(payload_with_crc)
    MAXLEN = len(random_mask_tuple)
    if L > MAXLEN:
        raise ValueError, "len(payload) must be in [0, %d]" % (MAXLEN,)

    if whitening:
        pkt = ''.join((packed_preamble, packed_access_code, make_header(L, whitener_offset),
                       whiten(payload_with_crc, whitener_offset), '\x55'))
    else:
        pkt = ''.join((packed_preamble, packed_access_code, make_header(L, whitener_offset),
                       (payload_with_crc), '\x55'))

    if pad_for_usrp:
        pkt = pkt + (_npadding_bytes(len(pkt), int(samples_per_symbol), bits_per_symbol) * '\x55')

    #print "make_packet: len(pkt) =", len(pkt)
    return pkt

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
    byte_modulus = gru.lcm(modulus/8, samples_per_symbol) * bits_per_symbol / samples_per_symbol
    r = pkt_byte_len % byte_modulus
    if r == 0:
        return 0
    return byte_modulus - r

# /////////////////////////////////////////////////////////////////////////////

if __name__ == '__main__':
	(s, padded) = conv_1_0_string_to_packed_binary_string("10101111")
	print "s=" + s
	(s, padded) = conv_1_0_string_to_packed_binary_string("1010111")
	print "s=" + s
	(s, padded) = conv_1_0_string_to_packed_binary_string("0101111")
	print "s=" + s
	(s, padded) = conv_1_0_string_to_packed_binary_string("1111010110101111")
	print "s=" + s
	(s, padded) = conv_1_0_string_to_packed_binary_string("111010110101111")
	print "s=" + s
	(s, padded) = conv_1_0_string_to_packed_binary_string("111101011010111")
	print "s=" + s
        s = conv_packed_binary_string_to_1_0_string("\xAC\xDD\xA4\xE2\xF2\x8C\x20\xFC")
        print "s=" + s
        s = conv_packed_binary_string_to_1_0_string("\xA4\xF2")
        print "s=" + s

        print conv_packed_binary_string_to_1_0_string(make_header(4095))

        dir(digital)
        #print make_packet("toast", 0, 0, default_access_code, False, 0, False)
