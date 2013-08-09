#!/usr/bin/env python

from gnuradio import digital
from gnuradio import eng_notation
from gnuradio import gr
from gnuradio.eng_option import eng_option
from gnuradio.gr import firdes
from optparse import OptionParser

from gnuradio.digital import digital_swig
import math

#from grc_gnuradio import blks2 as grc_blks2
import packet

import struct
import numpy
import sys

#from gnuradio import gr, digital
#from gnuradio.digital import packet_utils
import gnuradio.gr.gr_threading as _threading

#from gnuradio.digital import crc

from gnuradio.digital import packet_utils

from gnuradio import gru

# Default number of points in constellation.
_def_constellation_points = 2
# Whether differential coding is used.
_def_differential = False

_def_samples_per_symbol = 2
_def_excess_bw = 0.35
_def_verbose = False
_def_log = False

# Frequency correction
_def_freq_bw = 2*math.pi/100.0
# Symbol timing recovery 
_def_timing_bw = 2*math.pi/100.0
_def_timing_max_dev = 1.5
# Fine frequency / Phase correction
_def_phase_bw = 2*math.pi/100.0
# Number of points in constellation
_def_constellation_points = 16
# Whether differential coding is used.
_def_differential = False

##payload length in bytes
DEFAULT_PAYLOAD_LEN = 512

##how many messages in a queue
DEFAULT_MSGQ_LIMIT = 2

##threshold for unmaking packets
DEFAULT_THRESHOLD = 12

def gen_and_append_crc32(s):
    crc = digital_swig.crc32(s)
    return (s + struct.pack(">I", gru.hexint(crc) & 0xFFFFFFFF)), crc

def check_crc32(s):
    if len(s) < 4:
        return (False, '')
    msg = s[:-4]
    #print "msg = '%s'" % (msg,)
    actual = digital_swig.crc32(msg)
    (expected,) = struct.unpack(">I", s[-4:])
    # print "actual =", hex(actual), "expected =", hex(expected)
    return (actual == expected, msg, actual, expected)

def make_header(payload_len, whitener_offset=0):
    # Upper nibble is offset, lower 12 bits is len
    val = ((whitener_offset & 0xf) << 12) | (payload_len & 0x0fff)
    #print "offset =", whitener_offset, " len =", payload_len, " val=", val
    return struct.pack('!HH', val, val)

def make_packet(payload, samples_per_symbol, bits_per_symbol,
                access_code=packet_utils.default_access_code, pad_for_usrp=True,
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
    if not packet_utils.is_1_0_string(access_code):
        raise ValueError, "access_code must be a string containing only 0's and 1's (%r)" % (access_code,)

    if not whitener_offset >=0 and whitener_offset < 16:
        raise ValueError, "whitener_offset must be between 0 and 15, inclusive (%i)" % (whitener_offset,)

    (packed_access_code, padded) = packet_utils.conv_1_0_string_to_packed_binary_string(access_code)
    (packed_preamble, ignore) = packet_utils.conv_1_0_string_to_packed_binary_string(packet_utils.preamble)

    #print "make_packet:"
    #print payload.encode('hex').upper()
    
    payload_with_crc, crc = gen_and_append_crc32(payload)
    #print "outbound crc =", string_to_hex_list(payload_with_crc[-4:])

    L = len(payload_with_crc)
    MAXLEN = len(packet_utils.random_mask_tuple)
    if L > MAXLEN:
        raise ValueError, "len(payload) must be in [0, %d]" % (MAXLEN,)

    if whitening:
        pkt = ''.join((packed_preamble, packed_access_code, make_header(L, whitener_offset),
                       packet_utils.whiten(payload_with_crc, whitener_offset), '\x55'))
    else:
        pkt = ''.join((packed_preamble, packed_access_code, make_header(L, whitener_offset),
                       (payload_with_crc), '\x55'))

    if pad_for_usrp:
        pkt = pkt + (packet_utils._npadding_bytes(len(pkt), int(samples_per_symbol), bits_per_symbol) * '\x55')


    #sys.stderr.write("make_packet: crc=" + ("%.8X" % crc) + ", len=" + str(len(payload)) + "\n")
    #sys.stderr.write(payload[0:10].encode('hex').upper() + "\n")

    #print "make_packet: len(payload)=", len(payload), (", crc=%.8X" % crc), ", n_crc=", struct.pack(">I", crc).encode('hex').upper()
    #dewhitened_payload = packet_utils.whiten(payload_with_crc, whitener_offset)
    #print dewhitened_payload[0:10].encode('hex').upper()
    #print payload[0:10].encode('hex').upper()
    #print whitened_payload_with_crc.encode('hex').upper()
    return pkt

def unmake_packet(whitened_payload_with_crc, whitener_offset=0, dewhitening=True):
    """
    Return (ok, payload)

    @param whitened_payload_with_crc: string
    """

    if dewhitening:
        payload_with_crc = packet_utils.dewhiten(whitened_payload_with_crc, whitener_offset)
    else:
        payload_with_crc = (whitened_payload_with_crc)

    ok, payload, actual, expected = check_crc32(payload_with_crc)

    #sys.stderr.write("unmake_packet: expected=" + ("%.8X" % expected) + ", actual=" + ("%.8X" % actual) + ", ok=" + str(int(ok)) + "\n")
    #sys.stderr.write(payload[0:10].encode('hex').upper() + "\n")

    if 0:
        print "payload_with_crc =", string_to_hex_list(payload_with_crc)
        print "ok = %r, len(payload) = %d" % (ok, len(payload))
        print "payload =", string_to_hex_list(payload)

    return ok, payload

##################################################
## Options Class for OFDM
##################################################
class options(object):
	def __init__(self, **kwargs):
		for key, value in kwargs.iteritems(): setattr(self, key, value)

##################################################
## Packet Encoder
##################################################
class _packet_encoder_thread(_threading.Thread):

	def __init__(self, msgq, payload_length, send):
		self._msgq = msgq
		self._payload_length = payload_length
		self._send = send
		_threading.Thread.__init__(self)
		self.setDaemon(1)
		self.keep_running = True
		self.start()

	def run(self):
		sample = '' #residual sample
		while self.keep_running:
			msg = self._msgq.delete_head() #blocking read of message queue
			sample = sample + msg.to_string() #get the body of the msg as a string
			while len(sample) >= self._payload_length:
				payload = sample[:self._payload_length]
				sample = sample[self._payload_length:]
				self._send(payload)

class packet_encoder(gr.hier_block2):
	"""
	Hierarchical block for wrapping packet-based modulators.
	"""

	def __init__(self, samples_per_symbol, bits_per_symbol, access_code='', pad_for_usrp=True):
		"""
		packet_mod constructor.
		@param samples_per_symbol number of samples per symbol
		@param bits_per_symbol number of bits per symbol
		@param access_code AKA sync vector
		@param pad_for_usrp If true, packets are padded such that they end up a multiple of 128 samples
		@param payload_length number of bytes in a data-stream slice
		"""
		#setup parameters
		self._samples_per_symbol = samples_per_symbol
		self._bits_per_symbol = bits_per_symbol
		self._pad_for_usrp = pad_for_usrp
		if not access_code: #get access code
			access_code = packet_utils.default_access_code
		if not packet_utils.is_1_0_string(access_code):
			raise ValueError, "Invalid access_code %r. Must be string of 1's and 0's" % (access_code,)
		self._access_code = access_code
		self._pad_for_usrp = pad_for_usrp
		#create blocks
		msg_source = gr.message_source(gr.sizeof_char, DEFAULT_MSGQ_LIMIT)
		self._msgq_out = msg_source.msgq()
		#initialize hier2
		gr.hier_block2.__init__(
			self,
			"packet_encoder",
			gr.io_signature(0, 0, 0), # Input signature
			gr.io_signature(1, 1, gr.sizeof_char) # Output signature
		)
		#connect
		self.connect(msg_source, self)

	def send_pkt(self, payload):
		"""
		Wrap the payload in a packet and push onto the message queue.
		@param payload string, data to send
		"""
		packet = make_packet(
			payload,
			self._samples_per_symbol,
			self._bits_per_symbol,
			self._access_code,
			self._pad_for_usrp
		)
                #print payload[0:10].encode('hex').upper()
		msg = gr.message_from_string(packet)
		self._msgq_out.insert_tail(msg)

##################################################
## Packet Decoder
##################################################
class _packet_decoder_thread(_threading.Thread):

	def __init__(self, msgq, callback):
		_threading.Thread.__init__(self)
		self.setDaemon(1)
		self._msgq = msgq
		self.callback = callback
		self.keep_running = True
		self.start()

	def run(self):
		while self.keep_running:
			msg = self._msgq.delete_head()
			ok, payload = unmake_packet(msg.to_string(), int(msg.arg1()))
			if self.callback:
				self.callback(ok, payload)

class packet_decoder(gr.hier_block2):
	"""
	Hierarchical block for wrapping packet-based demodulators.
	"""

	def __init__(self, access_code='', threshold=-1, callback=None):
		"""
		packet_demod constructor.
		@param access_code AKA sync vector
		@param threshold detect access_code with up to threshold bits wrong (0 -> use default)
		@param callback a function of args: ok, payload
		"""
		#access code
		if not access_code: #get access code
			access_code = packet_utils.default_access_code
		if not packet_utils.is_1_0_string(access_code):
			raise ValueError, "Invalid access_code %r. Must be string of 1's and 0's" % (access_code,)
		self._access_code = access_code
		#threshold
		if threshold < 0: threshold = DEFAULT_THRESHOLD
		self._threshold = threshold
		#blocks
		msgq = gr.msg_queue(DEFAULT_MSGQ_LIMIT) #holds packets from the PHY
		correlator = digital.correlate_access_code_bb(self._access_code, self._threshold)
		framer_sink = gr.framer_sink_1(msgq)
		#initialize hier2
		gr.hier_block2.__init__(
			self,
			"packet_decoder",
			gr.io_signature(1, 1, gr.sizeof_char), # Input signature
			gr.io_signature(0, 0, 0) # Output signature
		)
                #test_block_1 = test_block()
		#connect
		self.connect(self, correlator, framer_sink)
		#start thread
		_packet_decoder_thread(msgq, callback)

##################################################
## Packet Mod for OFDM Mod and Packet Encoder
##################################################
class packet_mod_base(gr.hier_block2):
	"""
	Hierarchical block for wrapping packet source block.
	"""

	def __init__(self, packet_source=None, payload_length=0):
		if not payload_length: #get payload length
			payload_length = DEFAULT_PAYLOAD_LEN
		if payload_length%self._item_size_in != 0:	#verify that packet length is a multiple of the stream size
			raise ValueError, 'The payload length: "%d" is not a mutiple of the stream size: "%d".'%(payload_length, self._item_size_in)
		#initialize hier2
		gr.hier_block2.__init__(
			self,
			"ofdm_mod",
			gr.io_signature(1, 1, self._item_size_in), # Input signature
			gr.io_signature(1, 1, packet_source._hb.output_signature().sizeof_stream_item(0)) # Output signature
		)
		#create blocks
		msgq = gr.msg_queue(DEFAULT_MSGQ_LIMIT)
		msg_sink = gr.message_sink(self._item_size_in, msgq, False) #False -> blocking
		#connect
		self.connect(self, msg_sink)
		self.connect(packet_source, self)
		#start thread
		_packet_encoder_thread(msgq, payload_length, packet_source.send_pkt)

class packet_mod_b(packet_mod_base): _item_size_in = gr.sizeof_char
class packet_mod_s(packet_mod_base): _item_size_in = gr.sizeof_short
class packet_mod_i(packet_mod_base): _item_size_in = gr.sizeof_int
class packet_mod_f(packet_mod_base): _item_size_in = gr.sizeof_float
class packet_mod_c(packet_mod_base): _item_size_in = gr.sizeof_gr_complex

##################################################
## Packet Demod for OFDM Demod and Packet Decoder
##################################################
class packet_demod_base(gr.hier_block2):
	"""
	Hierarchical block for wrapping packet sink block.
	"""

	def __init__(self, packet_sink=None):
		#initialize hier2
		gr.hier_block2.__init__(
			self,
			"ofdm_mod",
			gr.io_signature(1, 1, packet_sink._hb.input_signature().sizeof_stream_item(0)), # Input signature
			gr.io_signature(1, 1, self._item_size_out) # Output signature
		)
		#create blocks
		msg_source = gr.message_source(self._item_size_out, DEFAULT_MSGQ_LIMIT)
		self._msgq_out = msg_source.msgq()
		#connect
		self.connect(self, packet_sink)
		self.connect(msg_source, self)
		if packet_sink._hb.output_signature().sizeof_stream_item(0):
			self.connect(packet_sink, gr.null_sink(packet_sink._hb.output_signature().sizeof_stream_item(0)))

	def recv_pkt(self, ok, payload):
		msg = gr.message_from_string(payload, 0, self._item_size_out, len(payload)/self._item_size_out)
		if ok: self._msgq_out.insert_tail(msg)

class packet_demod_b(packet_demod_base): _item_size_out = gr.sizeof_char
class packet_demod_s(packet_demod_base): _item_size_out = gr.sizeof_short
class packet_demod_i(packet_demod_base): _item_size_out = gr.sizeof_int
class packet_demod_f(packet_demod_base): _item_size_out = gr.sizeof_float
class packet_demod_c(packet_demod_base): _item_size_out = gr.sizeof_gr_complex

def invert_code(code):
    c = enumerate(code)
    ic = [(b, a) for (a, b) in c]
    ic.sort()
    return [a for (b, a) in ic]

class test_block(gr.sync_block):

    def __init__(self):
        gr.sync_block.__init__(self, name = "test_block",
			       in_sig = [numpy.uint8],
			       out_sig = [numpy.uint8])

    def work(self, input_items, output_items):
        n = min(len(input_items[0]), len(output_items[0]))

        sys.stdout.write("test_block.work: in=%d, out=%d, n=%d\n" % (len(input_items[0]), len(output_items[0]), n))

        for item in input_items[0][0:min(n, 80)]:
            sys.stdout.write("%02X" % item)
        sys.stdout.write("\n")

        output_items[0][:] = input_items[0][0:n]

        return n

class gr_complex_dump_cb(gr.basic_block):

    def __init__(self, precision):
        gr.basic_block.__init__(self, "complex_dump_cb",
                                [numpy.complex64],
                                [numpy.uint8])
        self.precision = precision
        self.msg_len = 2 * precision + 8;

    def general_work(self, input_items, output_items):
        n_in = len(input_items[0])
        n_out = len(output_items[0])
        consumed = 0
        produced = 0

        while consumed + 1 <= n_in and produced + self.msg_len <= n_out:
            if input_items[0][consumed].real < 0:
                real_sign = '-'
            else:
                real_sign = '+'

            if input_items[0][consumed].imag < 0:
                imag_sign = '-'
            else:
                imag_sign = '+'

            msg = "%c%.*f%ci%.*f\n" % (real_sign, self.precision, abs(input_items[0][consumed].real), imag_sign, self.precision, abs(input_items[0][consumed].imag))

            for i, c in enumerate(msg):
                output_items[0][produced + i] = ord(c)

            consumed = consumed + 1
            produced = produced + self.msg_len

        self.consume_each(consumed)

        return produced
        
    def forecast(self, noutput_items, ninput_items_required):
        ninput_items_required[0] = noutput_items / self.msg_len
        

# /////////////////////////////////////////////////////////////////////////////
#                             Generic modulator
# /////////////////////////////////////////////////////////////////////////////

class generic_mod(gr.hier_block2):

    def __init__(self, constellation,
                 samples_per_symbol=_def_samples_per_symbol,
                 differential=_def_differential,
                 excess_bw=_def_excess_bw,
                 gray_coded=True,
                 verbose=_def_verbose,
                 log=_def_log):
        """
	Hierarchical block for RRC-filtered differential generic modulation.

	The input is a byte stream (unsigned char) and the
	output is the complex modulated signal at baseband.
        
	@param constellation: determines the modulation type
	@type constellation: gnuradio.digital.gr_constellation
	@param samples_per_symbol: samples per baud >= 2
	@type samples_per_symbol: float
	@param excess_bw: Root-raised cosine filter excess bandwidth
	@type excess_bw: float
        @param gray_coded: turn gray coding on/off
        @type gray_coded: bool
        @param verbose: Print information about modulator?
        @type verbose: bool
        @param log: Log modulation data to files?
        @type log: bool
	"""

	gr.hier_block2.__init__(self, "generic_mod",
				gr.io_signature(1, 1, gr.sizeof_char),       # Input signature
				gr.io_signature(1, 1, gr.sizeof_gr_complex)) # Output signature

        self._constellation = constellation.base()
        self._samples_per_symbol = samples_per_symbol
        self._excess_bw = excess_bw
        self._differential = differential

        if self._samples_per_symbol < 2:
            raise TypeError, ("sbp must be >= 2, is %f" % self._samples_per_symbol)
        
        arity = pow(2,self.bits_per_symbol())
        
        # turn bytes into k-bit vectors
        self.bytes2chunks = \
          gr.packed_to_unpacked_bb(self.bits_per_symbol(), gr.GR_MSB_FIRST)

        if gray_coded == True:
            self.symbol_mapper = digital.map_bb(self._constellation.pre_diff_code())

        if differential:
            self.diffenc = digital.diff_encoder_bb(arity)

        self.chunks2symbols = digital.chunks_to_symbols_bc(self._constellation.points())

        # pulse shaping filter
        nfilts = 32
        ntaps = nfilts * 11 * int(self._samples_per_symbol)    # make nfilts filters of ntaps each
        self.rrc_taps = gr.firdes.root_raised_cosine(
            nfilts,          # gain
            nfilts,          # sampling rate based on 32 filters in resampler
            1.0,             # symbol rate
            self._excess_bw, # excess bandwidth (roll-off factor)
            ntaps)
        self.rrc_filter = gr.pfb_arb_resampler_ccf(self._samples_per_symbol,
                                                   self.rrc_taps, nfilts)
        
        #print "nfilts=%d, d_excess_bw=%.2f, ntaps=%d, samples_per_symbol=%.2f, rrc_taps=%d" % (nfilts, self._excess_bw, ntaps, self._samples_per_symbol, len(self.rrc_taps))

        #for tap in self.rrc_taps:
        #    sys.stdout.write("%.2f" % tap);
        #sys.stdout.write("\n");

        self.gr_file_sink_0 = gr.file_sink(gr.sizeof_char*1, "bytes2chunks2")
        self.gr_file_sink_0.set_unbuffered(True)
        self.gr_file_sink_1 = gr.file_sink(gr.sizeof_char*1, "symbol_mapper2")
        self.gr_file_sink_1.set_unbuffered(True)
        self.gr_file_sink_2 = gr.file_sink(gr.sizeof_char*1, "diffenc2")
        self.gr_file_sink_2.set_unbuffered(True)
        self.gr_file_sink_3 = gr.file_sink(gr.sizeof_gr_complex*1, "chunks2symbols2")
        self.gr_file_sink_3.set_unbuffered(True)
        self.gr_file_sink_4 = gr.file_sink(gr.sizeof_gr_complex*1, "rrc_filter2")
        self.gr_file_sink_4.set_unbuffered(True)

	# Connect
        blocks = [self, self.bytes2chunks]
        if gray_coded == True:
            blocks.append(self.symbol_mapper)
        if differential:
            blocks.append(self.diffenc)
        blocks += [self.chunks2symbols, self.rrc_filter, self]
        self.connect(*blocks)

        self.connect((self.bytes2chunks, 0), (self.gr_file_sink_0, 0))
        self.connect((self.symbol_mapper, 0), (self.gr_file_sink_1, 0))
        self.connect((self.diffenc, 0), (self.gr_file_sink_2, 0))
        self.connect((self.chunks2symbols, 0), (self.gr_file_sink_3, 0))
        self.connect((self.rrc_filter, 0), (self.gr_file_sink_4, 0))

        if verbose:
            self._print_verbage()
            
        if log:
            self._setup_logging()
            

    def samples_per_symbol(self):
        return self._samples_per_symbol

    def bits_per_symbol(self):   # static method that's also callable on an instance
        return self._constellation.bits_per_symbol()

    def add_options(parser):
        """
        Adds generic modulation options to the standard parser
        """
        add_common_options(parser)
    add_options=staticmethod(add_options)

    def extract_kwargs_from_options(cls, options):
        """
        Given command line options, create dictionary suitable for passing to __init__
        """
        return extract_kwargs_from_options_for_class(cls, options)
    extract_kwargs_from_options=classmethod(extract_kwargs_from_options)


    def _print_verbage(self):
        print "\nModulator:"
        print "bits per symbol:     %d" % self.bits_per_symbol()
        print "RRC roll-off factor: %.2f" % self._excess_bw

    def _setup_logging(self):
        print "Modulation logging turned on."
        self.connect(self.bytes2chunks,
                     gr.file_sink(gr.sizeof_char, "tx_bytes2chunks.8b"))
        if self._constellation.apply_pre_diff_code():
            self.connect(self.symbol_mapper,
                         gr.file_sink(gr.sizeof_char, "tx_symbol_mapper.8b"))
        if self._differential:
            self.connect(self.diffenc,
                         gr.file_sink(gr.sizeof_char, "tx_diffenc.8b"))
        self.connect(self.chunks2symbols,
                     gr.file_sink(gr.sizeof_gr_complex, "tx_chunks2symbols.32fc"))
        self.connect(self.rrc_filter,
                     gr.file_sink(gr.sizeof_gr_complex, "tx_rrc_filter.32fc"))

# /////////////////////////////////////////////////////////////////////////////
#                             Generic demodulator
#
#      Differentially coherent detection of differentially encoded generically
#      modulated signal.
# /////////////////////////////////////////////////////////////////////////////

class generic_demod(gr.hier_block2):

    def __init__(self, constellation,
                 samples_per_symbol=_def_samples_per_symbol,
                 differential=_def_differential,
                 excess_bw=_def_excess_bw,
                 gray_coded=True,
                 freq_bw=_def_freq_bw,
                 timing_bw=_def_timing_bw,
                 phase_bw=_def_phase_bw,
                 verbose=_def_verbose,
                 log=_def_log):
        """
	Hierarchical block for RRC-filtered differential generic demodulation.

	The input is the complex modulated signal at baseband.
	The output is a stream of bits packed 1 bit per byte (LSB)

	@param constellation: determines the modulation type
	@type constellation: gnuradio.digital.gr_constellation
	@param samples_per_symbol: samples per symbol >= 2
	@type samples_per_symbol: float
	@param excess_bw: Root-raised cosine filter excess bandwidth
	@type excess_bw: float
        @param gray_coded: turn gray coding on/off
        @type gray_coded: bool
        @param freq_bw: loop filter lock-in bandwidth
        @type freq_bw: float
        @param timing_bw: timing recovery loop lock-in bandwidth
        @type timing_bw: float
        @param phase_bw: phase recovery loop bandwidth
        @type phase_bw: float
        @param verbose: Print information about modulator?
        @type verbose: bool
        @param debug: Print modualtion data to files?
        @type debug: bool
	"""
        
	gr.hier_block2.__init__(self, "generic_demod",
				gr.io_signature(1, 1, gr.sizeof_gr_complex), # Input signature
				gr.io_signature(1, 1, gr.sizeof_char))       # Output signature
				
        self._constellation = constellation.base()
        self._samples_per_symbol = samples_per_symbol
        self._excess_bw = excess_bw
        self._phase_bw = phase_bw
        self._freq_bw = freq_bw
        self._timing_bw = timing_bw
        self._timing_max_dev= _def_timing_max_dev
        self._differential = differential

        if self._samples_per_symbol < 2:
            raise TypeError, ("sbp must be >= 2, is %d" % self._samples_per_symbol)

        arity = pow(2,self.bits_per_symbol())

        nfilts = 32
        ntaps = 11 * int(self._samples_per_symbol*nfilts)

        # Automatic gain control
        self.agc = gr.agc2_cc(0.6e-1, 1e-3, 1, 1, 100)

        # Frequency correction
        fll_ntaps = 55
        self.freq_recov = digital.fll_band_edge_cc(self._samples_per_symbol,
                                                   self._excess_bw,
                                                   fll_ntaps, self._freq_bw)

        # symbol timing recovery with RRC data filter
        taps = gr.firdes.root_raised_cosine(nfilts, nfilts*self._samples_per_symbol,
                                            1.0, self._excess_bw, ntaps)
        self.time_recov = digital.pfb_clock_sync_ccf(self._samples_per_symbol,
                                                     self._timing_bw, taps,
                                                     nfilts, nfilts/2,
                                                     self._timing_max_dev)

        fmin = -0.25
        fmax = 0.25
        self.receiver = digital.constellation_receiver_cb(
            self._constellation, self._phase_bw,
            fmin, fmax)

        # Do differential decoding based on phase change of symbols
        if differential:
            self.diffdec = digital.diff_decoder_bb(arity)

        if gray_coded:
            self.symbol_mapper = digital.map_bb(
                invert_code(self._constellation.pre_diff_code()))

        # unpack the k bit vector into a stream of bits
        self.unpack = gr.unpack_k_bits_bb(self.bits_per_symbol())

        if verbose:
            self._print_verbage()

        if log:
            self._setup_logging()

        self.gr_file_sink_0 = gr.file_sink(gr.sizeof_gr_complex*1, "agc2bin")
        self.gr_file_sink_0.set_unbuffered(True)
        self.gr_file_sink_1 = gr.file_sink(gr.sizeof_gr_complex*1, "freq_recov2bin")
        self.gr_file_sink_1.set_unbuffered(True)
        self.gr_file_sink_2 = gr.file_sink(gr.sizeof_gr_complex*1, "time_recov2bin")
        self.gr_file_sink_2.set_unbuffered(True)
        self.gr_file_sink_3 = gr.file_sink(gr.sizeof_char*1, "receiver2")
        self.gr_file_sink_3.set_unbuffered(True)
        self.gr_file_sink_4 = gr.file_sink(gr.sizeof_char*1, "diffdec2")
        self.gr_file_sink_4.set_unbuffered(True)
        self.gr_file_sink_5 = gr.file_sink(gr.sizeof_char*1, "demod_symbol_mapper2")
        self.gr_file_sink_5.set_unbuffered(True)
        self.gr_file_sink_6 = gr.file_sink(gr.sizeof_char*1, "unpack2")
        self.gr_file_sink_6.set_unbuffered(True)

        #self.gr_file_sink_7 = gr.file_sink(gr.sizeof_char*1, "agc2")
        #self.gr_file_sink_7.set_unbuffered(True)

        #complex_dump_0 = gr_complex_dump_cb(4)
        #complex_dump_1 = gr_complex_dump_cb(4)
        #complex_dump_2 = gr_complex_dump_cb(4)
        
        # Connect and Initialize base class
        blocks = [self, self.agc, self.freq_recov,
                  self.time_recov, self.receiver]
        if differential:
            blocks.append(self.diffdec)
        if self._constellation.apply_pre_diff_code():
            blocks.append(self.symbol_mapper)
        blocks += [self.unpack, self]
        self.connect(*blocks)

        #self.connect((self.agc, 0), (complex_dump_0, 0))
        #self.connect((complex_dump_0, 0), (self.gr_file_sink_7, 0))
        self.connect((self.agc, 0), (self.gr_file_sink_0, 0))
        #self.connect((self.freq_recov, 0), (complex_dump_1, 0))
        #self.connect((complex_dump_1, 0), (self.gr_file_sink_1, 0))
        self.connect((self.freq_recov, 0), (self.gr_file_sink_1, 0))
        #self.connect((self.time_recov, 0), (complex_dump_2, 0))
        #self.connect((complex_dump_2, 0), (self.gr_file_sink_2, 0))
        self.connect((self.time_recov, 0), (self.gr_file_sink_2, 0))
        self.connect((self.receiver, 0), (self.gr_file_sink_3, 0))
        self.connect((self.diffdec, 0), (self.gr_file_sink_4, 0))
        if self._constellation.apply_pre_diff_code():
            self.connect((self.symbol_mapper, 0), (self.gr_file_sink_5, 0))
        self.connect((self.unpack, 0), (self.gr_file_sink_6, 0))

    def samples_per_symbol(self):
        return self._samples_per_symbol

    def bits_per_symbol(self):   # staticmethod that's also callable on an instance
        return self._constellation.bits_per_symbol()

    def _print_verbage(self):
        print "\nDemodulator:"
        print "bits per symbol:     %d"   % self.bits_per_symbol()
        print "RRC roll-off factor: %.2f" % self._excess_bw
        print "FLL bandwidth:       %.2e" % self._freq_bw
        print "Timing bandwidth:    %.2e" % self._timing_bw
        print "Phase bandwidth:     %.2e" % self._phase_bw

    def _setup_logging(self):
        print "Modulation logging turned on."
        self.connect(self.agc,
                     gr.file_sink(gr.sizeof_gr_complex, "rx_agc.32fc"))
        self.connect((self.freq_recov, 0),
                     gr.file_sink(gr.sizeof_gr_complex, "rx_freq_recov.32fc"))
        self.connect((self.freq_recov, 1),
                     gr.file_sink(gr.sizeof_float, "rx_freq_recov_freq.32f"))
        self.connect((self.freq_recov, 2),
                     gr.file_sink(gr.sizeof_float, "rx_freq_recov_phase.32f"))
        self.connect((self.freq_recov, 3),
                     gr.file_sink(gr.sizeof_float, "rx_freq_recov_error.32f"))
        self.connect((self.time_recov, 0),
                     gr.file_sink(gr.sizeof_gr_complex, "rx_time_recov.32fc"))
        self.connect((self.time_recov, 1),
                     gr.file_sink(gr.sizeof_float, "rx_time_recov_error.32f"))
        self.connect((self.time_recov, 2),
                     gr.file_sink(gr.sizeof_float, "rx_time_recov_rate.32f"))
        self.connect((self.time_recov, 3),
                     gr.file_sink(gr.sizeof_float, "rx_time_recov_phase.32f"))
        self.connect((self.receiver, 0),
                     gr.file_sink(gr.sizeof_char, "rx_receiver.8b"))
        self.connect((self.receiver, 1),
                     gr.file_sink(gr.sizeof_float, "rx_receiver_error.32f"))
        self.connect((self.receiver, 2),
                     gr.file_sink(gr.sizeof_float, "rx_receiver_phase.32f"))
        self.connect((self.receiver, 3),
                     gr.file_sink(gr.sizeof_float, "rx_receiver_freq.32f"))
        if self._differential:
            self.connect(self.diffdec,
                         gr.file_sink(gr.sizeof_char, "rx_diffdec.8b"))
        if self._constellation.apply_pre_diff_code():
            self.connect(self.symbol_mapper,
                         gr.file_sink(gr.sizeof_char, "rx_symbol_mapper.8b"))
        self.connect(self.unpack,
                     gr.file_sink(gr.sizeof_char, "rx_unpack.8b"))
        
    def add_options(parser):
        """
        Adds generic demodulation options to the standard parser
        """
        # Add options shared with modulator.
        add_common_options(parser)
        # Add options specific to demodulator.
        parser.add_option("", "--freq-bw", type="float", default=_def_freq_bw,
                          help="set frequency lock loop lock-in bandwidth [default=%default]")
        parser.add_option("", "--phase-bw", type="float", default=_def_phase_bw,
                          help="set phase tracking loop lock-in bandwidth [default=%default]")
        parser.add_option("", "--timing-bw", type="float", default=_def_timing_bw,
                          help="set timing symbol sync loop gain lock-in bandwidth [default=%default]")
    add_options=staticmethod(add_options)
    
    def extract_kwargs_from_options(cls, options):
        """
        Given command line options, create dictionary suitable for passing to __init__
        """
        return extract_kwargs_from_options_for_class(cls, options)
    extract_kwargs_from_options=classmethod(extract_kwargs_from_options)

# /////////////////////////////////////////////////////////////////////////////

# FYI, this PN code is the output of a 15-bit LFSR
random_mask_tuple = (
  255,  63,   0,  16,   0,  12,   0,   5, 192,   3,  16,   1, 204,   0,  85, 192, 
   63,  16,  16,  12,  12,   5, 197, 195,  19,  17, 205, 204,  85, 149, 255,  47, 
    0,  28,   0,   9, 192,   6, 208,   2, 220,   1, 153, 192, 106, 208,  47,  28, 
   28,   9, 201, 198, 214, 210, 222, 221, 152,  89, 170, 186, 255,  51,   0,  21, 
  192,  15,  16,   4,  12,   3,  69, 193, 243,  16,  69, 204,  51,  21, 213, 207, 
   31,  20,   8,  15,  70, 132,  50, 227,  85, 137, 255,  38, 192,  26, 208,  11, 
   28,   7,  73, 194, 182, 209, 182, 220, 118, 217, 230, 218, 202, 219,  23,  27, 
   78, 139, 116, 103, 103, 106, 170, 175,  63,  60,  16,  17, 204,  12,  85, 197, 
  255,  19,   0,  13, 192,   5, 144,   3,  44,   1, 221, 192,  89, 144,  58, 236, 
   19,  13, 205, 197, 149, 147,  47,  45, 220,  29, 153, 201, 170, 214, 255,  30, 
  192,   8,  80,   6, 188,   2, 241, 193, 132,  80,  99, 124,  41, 225, 222, 200, 
   88,  86, 186, 190, 243,  48,  69, 212,  51,  31,  85, 200,  63,  22, 144,  14, 
  236,   4,  77, 195, 117, 145, 231,  44,  74, 157, 247,  41, 134, 158, 226, 232, 
   73, 142, 182, 228, 118, 203, 102, 215, 106, 222, 175,  24, 124,  10, 161, 199, 
   56,  82, 146, 189, 173, 177, 189, 180, 113, 183, 100, 118, 171, 102, 255, 106, 
  192,  47,  16,  28,  12,   9, 197, 198, 211,  18, 221, 205, 153, 149, 170, 239, 
   63,  12,  16,   5, 204,   3,  21, 193, 207,  16,  84,  12,  63,  69, 208,  51, 
   28,  21, 201, 207,  22, 212,  14, 223,  68,  88,  51, 122, 149, 227,  47,   9, 
  220,   6, 217, 194, 218, 209, 155,  28, 107,  73, 239, 118, 204,  38, 213, 218, 
  223,  27,  24,  11,  74, 135, 119,  34, 166, 153, 186, 234, 243,  15,   5, 196, 
    3,  19,  65, 205, 240,  85, 132,  63,  35,  80,  25, 252,  10, 193, 199,  16, 
   82, 140,  61, 165, 209, 187,  28, 115,  73, 229, 246, 203,   6, 215,  66, 222, 
  177, 152, 116, 106, 167, 111,  58, 172,  19,  61, 205, 209, 149, 156, 111,  41, 
  236,  30, 205, 200,  85, 150, 191,  46, 240,  28,  68,   9, 243,  70, 197, 242, 
  211,   5, 157, 195,  41, 145, 222, 236,  88,  77, 250, 181, 131,  55,  33, 214, 
  152,  94, 234, 184,  79,  50, 180,  21, 183,  79,  54, 180,  22, 247,  78, 198, 
  180,  82, 247, 125, 134, 161, 162, 248, 121, 130, 162, 225, 185, 136, 114, 230, 
  165, 138, 251,  39,   3,  90, 129, 251,  32,  67,  88,  49, 250, 148,  67,  47, 
  113, 220,  36,  89, 219, 122, 219,  99,  27, 105, 203, 110, 215, 108,  94, 173, 
  248, 125, 130, 161, 161, 184, 120, 114, 162, 165, 185, 187,  50, 243,  85, 133, 
  255,  35,   0,  25, 192,  10, 208,   7,  28,   2, 137, 193, 166, 208, 122, 220, 
   35,  25, 217, 202, 218, 215,  27,  30, 139,  72, 103, 118, 170, 166, 255,  58, 
  192,  19,  16,  13, 204,   5, 149, 195,  47,  17, 220,  12,  89, 197, 250, 211, 
    3,  29, 193, 201, 144,  86, 236,  62, 205, 208,  85, 156,  63,  41, 208,  30, 
  220,   8,  89, 198, 186, 210, 243,  29, 133, 201, 163,  22, 249, 206, 194, 212, 
   81, 159, 124, 104,  33, 238, 152,  76, 106, 181, 239,  55,  12,  22, 133, 206, 
  227,  20,  73, 207, 118, 212,  38, 223,  90, 216,  59,  26, 147,  75,  45, 247, 
   93, 134, 185, 162, 242, 249, 133, 130, 227,  33, 137, 216, 102, 218, 170, 219, 
   63,  27,  80,  11, 124,   7,  97, 194, 168,  81, 190, 188, 112, 113, 228,  36, 
   75,  91, 119, 123, 102, 163, 106, 249, 239,   2, 204,   1, 149, 192, 111,  16, 
   44,  12,  29, 197, 201, 147,  22, 237, 206, 205, 148,  85, 175, 127,  60,  32, 
   17, 216,  12,  90, 133, 251,  35,   3,  89, 193, 250, 208,  67,  28,  49, 201, 
  212,  86, 223, 126, 216,  32,  90, 152,  59,  42, 147,  95,  45, 248,  29, 130, 
  137, 161, 166, 248, 122, 194, 163,  17, 185, 204, 114, 213, 229, 159,  11,  40, 
    7,  94, 130, 184,  97, 178, 168, 117, 190, 167,  48, 122, 148,  35,  47,  89, 
  220,  58, 217, 211,  26, 221, 203,  25, 151,  74, 238, 183,  12, 118, 133, 230, 
  227,  10, 201, 199,  22, 210, 142, 221, 164,  89, 187, 122, 243,  99,   5, 233, 
  195,  14, 209, 196,  92,  83, 121, 253, 226, 193, 137, 144, 102, 236,  42, 205, 
  223,  21, 152,  15,  42, 132,  31,  35,  72,  25, 246, 138, 198, 231,  18, 202, 
  141, 151,  37, 174, 155,  60, 107,  81, 239, 124,  76,  33, 245, 216,  71,  26, 
  178, 139,  53, 167,  87,  58, 190, 147,  48, 109, 212,  45, 159,  93, 168,  57, 
  190, 146, 240, 109, 132,  45, 163,  93, 185, 249, 178, 194, 245, 145, 135,  44, 
   98, 157, 233, 169, 142, 254, 228,  64,  75, 112,  55, 100,  22, 171,  78, 255, 
  116,  64,  39, 112,  26, 164,  11,  59,  71,  83, 114, 189, 229, 177, 139,  52, 
  103,  87, 106, 190, 175,  48, 124,  20,  33, 207,  88,  84,  58, 191,  83,  48, 
   61, 212,  17, 159,  76, 104,  53, 238, 151,  12, 110, 133, 236,  99,  13, 233, 
  197, 142, 211,  36,  93, 219, 121, 155,  98, 235, 105, 143, 110, 228,  44,  75, 
   93, 247, 121, 134, 162, 226, 249, 137, 130, 230, 225, 138, 200, 103,  22, 170, 
  142, 255,  36,  64,  27, 112,  11, 100,   7, 107,  66, 175, 113, 188,  36, 113, 
  219, 100,  91, 107, 123, 111,  99, 108,  41, 237, 222, 205, 152,  85, 170, 191, 
   63,  48,  16,  20,  12,  15,  69, 196,  51,  19,  85, 205, 255,  21, 128,  15, 
   32,   4,  24,   3,  74, 129, 247,  32,  70, 152,  50, 234, 149, 143,  47,  36, 
   28,  27,  73, 203, 118, 215, 102, 222, 170, 216, 127,  26, 160,  11,  56,   7, 
   82, 130, 189, 161, 177, 184, 116, 114, 167, 101, 186, 171,  51,  63,  85, 208, 
   63,  28,  16,   9, 204,   6, 213, 194, 223,  17, 152,  12, 106, 133, 239,  35, 
   12,  25, 197, 202, 211,  23,  29, 206, 137, 148, 102, 239, 106, 204,  47,  21, 
  220,  15,  25, 196,  10, 211,  71,  29, 242, 137, 133, 166, 227,  58, 201, 211, 
   22, 221, 206, 217, 148,  90, 239, 123,  12,  35,  69, 217, 243,  26, 197, 203, 
   19,  23,  77, 206, 181, 148, 119,  47, 102, 156,  42, 233, 223,  14, 216,   4, 
   90, 131, 123,  33, 227,  88,  73, 250, 182, 195,  54, 209, 214, 220,  94, 217, 
  248,  90, 194, 187,  17, 179,  76, 117, 245, 231,   7,  10, 130, 135,  33, 162, 
  152, 121, 170, 162, 255,  57, 128,  18, 224,  13, 136,   5, 166, 131,  58, 225, 
  211,   8,  93, 198, 185, 146, 242, 237, 133, 141, 163,  37, 185, 219,  50, 219, 
   85, 155, 127,  43,  96,  31, 104,   8,  46, 134, 156,  98, 233, 233, 142, 206, 
  228,  84,  75, 127, 119,  96,  38, 168,  26, 254, 139,   0, 103,  64,  42, 176, 
   31,  52,   8,  23,  70, 142, 178, 228, 117, 139, 103,  39, 106, 154, 175,  43, 
   60,  31,  81, 200,  60,  86, 145, 254, 236,  64,  77, 240,  53, 132,  23,  35, 
   78, 153, 244, 106, 199, 111,  18, 172,  13, 189, 197, 177, 147,  52, 109, 215, 
  109, 158, 173, 168, 125, 190, 161, 176, 120, 116,  34, 167,  89, 186, 186, 243, 
   51,   5, 213, 195,  31,  17, 200,  12,  86, 133, 254, 227,   0,  73, 192,  54, 
  208,  22, 220,  14, 217, 196,  90, 211, 123,  29, 227,  73, 137, 246, 230, 198, 
  202, 210, 215,  29, 158, 137, 168, 102, 254, 170, 192, 127,  16,  32,  12,  24, 
    5, 202, 131,  23,  33, 206, 152,  84, 106, 191, 111,  48,  44,  20,  29, 207, 
   73, 148,  54, 239,  86, 204,  62, 213, 208,  95,  28,  56,   9, 210, 134, 221, 
  162, 217, 185, 154, 242, 235,   5, 143,  67,  36,  49, 219,  84,  91, 127, 123, 
   96,  35, 104,  25, 238, 138, 204, 103,  21, 234, 143,  15,  36,   4,  27,  67, 
   75, 113, 247, 100,  70, 171, 114, 255, 101, 128,  43,  32,  31,  88,   8,  58, 
  134, 147,  34, 237, 217, 141, 154, 229, 171,  11,  63,  71,  80,  50, 188,  21, 
  177, 207,  52,  84,  23, 127,  78, 160,  52, 120,  23,  98, 142, 169, 164, 126, 
  251,  96,  67, 104,  49, 238, 148,  76, 111, 117, 236,  39,  13, 218, 133, 155, 
   35,  43,  89, 223, 122, 216,  35,  26, 153, 203,  42, 215,  95,  30, 184,   8, 
  114, 134, 165, 162, 251,  57, 131,  82, 225, 253, 136,  65, 166, 176, 122, 244, 
   35,   7,  89, 194, 186, 209, 179,  28, 117, 201, 231,  22, 202, 142, 215,  36, 
   94, 155, 120, 107,  98, 175, 105, 188,  46, 241, 220,  68,  89, 243, 122, 197, 
  227,  19,   9, 205, 198, 213, 146, 223,  45, 152,  29, 170, 137, 191,  38, 240, 
   26, 196,  11,  19,  71,  77, 242, 181, 133, 183,  35,  54, 153, 214, 234, 222, 
  207,  24,  84,  10, 191,  71,  48,  50, 148,  21, 175,  79,  60,  52,  17, 215, 
   76,  94, 181, 248, 119,   2, 166, 129, 186, 224, 115,   8,  37, 198, 155,  18, 
  235,  77, 143, 117, 164,  39,  59,  90, 147, 123,  45, 227,  93, 137, 249, 166, 
  194, 250, 209, 131,  28,  97, 201, 232,  86, 206, 190, 212, 112,  95, 100,  56, 
   43,  82, 159, 125, 168,  33, 190, 152, 112, 106, 164,  47,  59,  92,  19, 121, 
  205, 226, 213, 137, 159,  38, 232,  26, 206, 139,  20, 103,  79, 106, 180,  47, 
   55,  92,  22, 185, 206, 242, 212,  69, 159, 115,  40,  37, 222, 155,  24, 107, 
   74, 175, 119,  60,  38, 145, 218, 236,  91,  13, 251,  69, 131, 115,  33, 229, 
  216,  75,  26, 183,  75,  54, 183,  86, 246, 190, 198, 240,  82, 196,  61, 147, 
   81, 173, 252, 125, 129, 225, 160,  72, 120,  54, 162, 150, 249, 174, 194, 252, 
   81, 129, 252,  96,  65, 232,  48,  78, 148,  52, 111,  87, 108,  62, 173, 208, 
  125, 156,  33, 169, 216, 126, 218, 160,  91,  56,  59,  82, 147, 125, 173, 225, 
  189, 136, 113, 166, 164, 122, 251,  99,   3, 105, 193, 238, 208,  76,  92,  53, 
  249, 215,   2, 222, 129, 152,  96, 106, 168,  47,  62, 156,  16, 105, 204,  46, 
  213, 220,  95,  25, 248,  10, 194, 135,  17, 162, 140, 121, 165, 226, 251,   9, 
  131,  70, 225, 242, 200,  69, 150, 179,  46, 245, 220,  71,  25, 242, 138, 197, 
  167,  19,  58, 141, 211,  37, 157, 219,  41, 155,  94, 235, 120,  79,  98, 180, 
   41, 183,  94, 246, 184,  70, 242, 178, 197, 181, 147,  55,  45, 214, 157, 158, 
  233, 168,  78, 254, 180,  64, 119, 112,  38, 164,  26, 251,  75,   3, 119,  65, 
  230, 176,  74, 244,  55,   7,  86, 130, 190, 225, 176,  72, 116,  54, 167,  86, 
  250, 190, 195,  48,  81, 212,  60,  95,  81, 248,  60,  66, 145, 241, 172,  68, 
  125, 243,  97, 133, 232,  99,  14, 169, 196, 126, 211,  96,  93, 232,  57, 142, 
  146, 228, 109, 139, 109, 167, 109, 186, 173, 179,  61, 181, 209, 183,  28, 118, 
  137, 230, 230, 202, 202, 215,  23,  30, 142, 136, 100, 102, 171, 106, 255, 111, 
    0,  44,   0,  29, 192,   9, 144,   6, 236,   2, 205, 193, 149, 144, 111,  44, 
   44,  29, 221, 201, 153, 150, 234, 238, 207,  12,  84,   5, 255,  67,   0,  49, 
  192,  20,  80,  15, 124,   4,  33, 195,  88,  81, 250, 188,  67,  49, 241, 212, 
   68,  95, 115, 120,  37, 226, 155,   9, 171,  70, 255, 114, 192,  37, 144,  27, 
   44,  11,  93, 199, 121, 146, 162, 237, 185, 141, 178, 229, 181, 139,  55,  39, 
   86, 154, 190, 235,  48,  79,  84,  52,  63,  87,  80,  62, 188,  16, 113, 204, 
   36,  85, 219, 127,  27,  96,  11, 104,   7, 110, 130, 172,  97, 189, 232, 113, 
  142, 164, 100, 123, 107,  99, 111, 105, 236,  46, 205, 220,  85, 153, 255,  42, 
  192,  31,  16,   8,  12,   6, 133, 194, 227,  17, 137, 204, 102, 213, 234, 223, 
   15,  24,   4,  10, 131,  71,  33, 242, 152,  69, 170, 179,  63,  53, 208,  23, 
   28,  14, 137, 196, 102, 211, 106, 221, 239,  25, 140,  10, 229, 199,  11,  18, 
  135,  77, 162, 181, 185, 183,  50, 246, 149, 134, 239,  34, 204,  25, 149, 202, 
  239,  23,  12,  14, 133, 196,  99,  19, 105, 205, 238, 213, 140,  95,  37, 248, 
   27,   2, 139,  65, 167, 112, 122, 164,  35,  59,  89, 211, 122, 221, 227,  25, 
  137, 202, 230, 215,  10, 222, 135,  24,  98, 138, 169, 167,  62, 250, 144,  67, 
   44,  49, 221, 212,  89, 159, 122, 232,  35,  14, 153, 196, 106, 211, 111,  29, 
  236,   9, 141, 198, 229, 146, 203,  45, 151,  93, 174, 185, 188, 114, 241, 229, 
  132,  75,  35, 119,  89, 230, 186, 202, 243,  23,   5, 206, 131,  20,  97, 207, 
  104,  84,  46, 191,  92, 112,  57, 228,  18, 203,  77, 151, 117, 174, 167,  60, 
  122, 145, 227,  44,  73, 221, 246, 217, 134, 218, 226, 219,   9, 155,  70, 235, 
  114, 207, 101, 148,  43,  47,  95,  92,  56,  57, 210, 146, 221, 173, 153, 189, 
  170, 241, 191,   4, 112,   3, 100,   1, 235,  64,  79, 112,  52,  36,  23,  91, 
   78, 187, 116, 115, 103, 101, 234, 171,  15,  63,  68,  16,  51,  76,  21, 245, 
  207,   7,  20,   2, 143,  65, 164,  48, 123,  84,  35, 127,  89, 224,  58, 200, 
   19,  22, 141, 206, 229, 148,  75,  47, 119,  92,  38, 185, 218, 242, 219,   5, 
  155,  67,  43, 113, 223, 100,  88,  43, 122, 159,  99,  40,  41, 222, 158, 216, 
  104,  90, 174, 187,  60, 115,  81, 229, 252,  75,   1, 247,  64,  70, 176,  50, 
  244,  21, 135,  79,  34, 180,  25, 183,  74, 246, 183,   6, 246, 130, 198, 225, 
  146, 200, 109, 150, 173, 174, 253, 188,  65, 177, 240, 116,  68,  39, 115,  90, 
  165, 251,  59,   3,  83,  65, 253, 240,  65, 132,  48,  99,  84,  41, 255,  94, 
  192,  56,  80,  18, 188,  13, 177, 197, 180,  83,  55, 125, 214, 161, 158, 248, 
  104,  66, 174, 177, 188, 116, 113, 231, 100,  74, 171, 119,  63, 102, 144,  42, 
  236,  31,  13, 200,   5, 150, 131,  46, 225, 220,  72,  89, 246, 186, 198, 243, 
   18, 197, 205, 147,  21, 173, 207,  61, 148,  17, 175,  76, 124,  53, 225, 215, 
    8,  94, 134, 184,  98, 242, 169, 133, 190, 227,  48,  73, 212,  54, 223,  86, 
  216,  62, 218, 144,  91,  44,  59,  93, 211, 121, 157, 226, 233, 137, 142, 230, 
  228,  74, 203, 119,  23, 102, 142, 170, 228, 127,  11,  96,   7, 104,   2, 174, 
  129, 188,  96, 113, 232,  36,  78, 155, 116, 107, 103, 111, 106, 172,  47,  61, 
  220,  17, 153, 204, 106, 213, 239,  31,  12,   8,   5, 198, 131,  18, 225, 205, 
  136,  85, 166, 191,  58, 240,  19,   4,  13, 195,  69, 145, 243,  44,  69, 221, 
  243,  25, 133, 202, 227,  23,   9, 206, 134, 212,  98, 223, 105, 152,  46, 234, 
  156,  79,  41, 244,  30, 199,  72,  82, 182, 189, 182, 241, 182, 196, 118, 211, 
  102, 221, 234, 217, 143,  26, 228,  11,  11,  71,  71, 114, 178, 165, 181, 187, 
   55,  51,  86, 149, 254, 239,   0,  76,   0,  53, 192,  23,  16,  14, 140,   4, 
  101, 195, 107,  17, 239,  76,  76,  53, 245, 215,   7,  30, 130, 136,  97, 166, 
  168, 122, 254, 163,   0, 121, 192,  34, 208,  25, 156,  10, 233, 199,  14, 210, 
  132,  93, 163, 121, 185, 226, 242, 201, 133, 150, 227,  46, 201, 220,  86, 217, 
  254, 218, 192,  91,  16,  59,  76,  19, 117, 205, 231,  21, 138, 143,  39,  36, 
   26, 155,  75,  43, 119,  95, 102, 184,  42, 242, 159,   5, 168,   3,  62, 129, 
  208,  96,  92,  40,  57, 222, 146, 216, 109, 154, 173, 171,  61, 191,  81, 176, 
   60, 116,  17, 231,  76,  74, 181, 247,  55,   6, 150, 130, 238, 225, 140,  72, 
  101, 246, 171,   6, 255,  66, 192,  49, 144,  20, 108,  15, 109, 196,  45, 147, 
   93, 173, 249, 189, 130, 241, 161, 132, 120,  99,  98, 169, 233, 190, 206, 240, 
   84,  68,  63, 115,  80,  37, 252,  27,   1, 203,  64,  87, 112,  62, 164,  16, 
  123,  76,  35, 117, 217, 231,  26, 202, 139,  23,  39,  78, 154, 180, 107,  55, 
  111,  86, 172,  62, 253, 208,  65, 156,  48, 105, 212,  46, 223,  92,  88,  57, 
  250, 146, 195,  45, 145, 221, 172,  89, 189, 250, 241, 131,   4,  97, 195, 104, 
   81, 238, 188,  76, 113, 245, 228,  71,  11, 114, 135, 101, 162, 171,  57, 191, 
   82, 240,  61, 132,  17, 163,  76, 121, 245, 226, 199,   9, 146, 134, 237, 162, 
  205, 185, 149, 178, 239,  53, 140,  23,  37, 206, 155,  20, 107,  79, 111, 116, 
   44,  39,  93, 218, 185, 155,  50, 235,  85, 143, 127,  36,  32,  27,  88,  11, 
  122, 135,  99,  34, 169, 217, 190, 218, 240,  91,   4,  59,  67,  83, 113, 253, 
  228,  65, 139, 112, 103, 100,  42, 171,  95,  63, 120,  16,  34, 140,  25, 165, 
  202, 251,  23,   3,  78, 129, 244,  96,  71, 104,  50, 174, 149, 188, 111,  49, 
  236,  20,  77, 207, 117, 148,  39,  47,  90, 156,  59,  41, 211,  94, 221, 248, 
   89, 130, 186, 225, 179,   8, 117, 198, 167,  18, 250, 141, 131,  37, 161, 219, 
   56,  91,  82, 187, 125, 179,  97, 181, 232, 119,  14, 166, 132, 122, 227,  99, 
    9, 233, 198, 206, 210, 212,  93, 159, 121, 168,  34, 254, 153, 128, 106, 224, 
   47,   8,  28,   6, 137, 194, 230, 209, 138, 220, 103,  25, 234, 138, 207,  39, 
   20,  26, 143,  75,  36,  55,  91,  86, 187, 126, 243,  96,  69, 232,  51,  14, 
  149, 196, 111,  19, 108,  13, 237, 197, 141, 147,  37, 173, 219,  61, 155,  81, 
  171, 124, 127,  97, 224,  40,  72,  30, 182, 136, 118, 230, 166, 202, 250, 215, 
    3,  30, 129, 200,  96,  86, 168,  62, 254, 144,  64, 108,  48,  45, 212,  29, 
  159,  73, 168,  54, 254, 150, 192, 110, 208,  44,  92,  29, 249, 201, 130, 214, 
  225, 158, 200, 104,  86, 174, 190, 252, 112,  65, 228,  48,  75,  84,  55, 127, 
   86, 160,  62, 248,  16,  66, 140,  49, 165, 212, 123,  31,  99,  72,  41, 246, 
  158, 198, 232,  82, 206, 189, 148, 113, 175, 100, 124,  43,  97, 223, 104,  88, 
   46, 186, 156, 115,  41, 229, 222, 203,  24,  87,  74, 190, 183,  48, 118, 148, 
   38, 239,  90, 204,  59,  21, 211,  79,  29, 244,   9, 135,  70, 226, 178, 201, 
  181, 150, 247,  46, 198, 156,  82, 233, 253, 142, 193, 164,  80, 123, 124,  35, 
   97, 217, 232,  90, 206, 187,  20, 115,  79, 101, 244,  43,   7,  95,  66, 184, 
   49, 178, 148, 117, 175, 103,  60,  42, 145, 223,  44,  88,  29, 250, 137, 131, 
   38, 225, 218, 200,  91,  22, 187,  78, 243, 116,  69, 231, 115,  10, 165, 199, 
   59,  18, 147,  77, 173, 245, 189, 135,  49, 162, 148, 121, 175,  98, 252,  41, 
  129, 222, 224,  88,  72,  58, 182, 147,  54, 237, 214, 205, 158, 213, 168,  95, 
   62, 184,  16, 114, 140,  37, 165, 219,  59,  27,  83,  75, 125, 247,  97, 134, 
  168,  98, 254, 169, 128, 126, 224,  32,  72,  24,  54, 138, 150, 231,  46, 202, 
  156,  87,  41, 254, 158, 192, 104,  80,  46, 188,  28, 113, 201, 228,  86, 203, 
  126, 215,  96,  94, 168,  56, 126, 146, 160, 109, 184,  45, 178, 157, 181, 169, 
  183,  62, 246, 144,  70, 236,  50, 205, 213, 149, 159,  47,  40,  28,  30, 137, 
  200, 102, 214, 170, 222, 255,  24,  64,  10, 176,   7,  52,   2, 151,  65, 174, 
  176, 124, 116,  33, 231,  88,  74, 186, 183,  51,  54, 149, 214, 239,  30, 204, 
    8,  85, 198, 191,  18, 240,  13, 132,   5, 163,  67,  57, 241, 210, 196,  93, 
  147, 121, 173, 226, 253, 137, 129, 166, 224, 122, 200,  35,  22, 153, 206, 234, 
  212,  79,  31, 116,   8,  39,  70, 154, 178, 235,  53, 143,  87,  36,  62, 155, 
   80, 107, 124,  47,  97, 220,  40,  89, 222, 186, 216, 115,  26, 165, 203,  59, 
   23,  83,  78, 189, 244, 113, 135, 100,  98, 171, 105, 191, 110, 240,  44,  68, 
   29, 243,  73, 133, 246, 227,   6, 201, 194, 214, 209, 158, 220, 104,  89, 238, 
  186, 204, 115,  21, 229, 207,  11,  20,   7,  79,  66, 180,  49, 183,  84, 118, 
  191, 102, 240,  42, 196,  31,  19,  72,  13, 246, 133, 134, 227,  34, 201, 217, 
  150, 218, 238, 219,  12,  91,  69, 251, 115,   3, 101, 193, 235,  16,  79,  76, 
   52,  53, 215,  87,  30, 190, 136, 112, 102, 164,  42, 251,  95,   3, 120,   1, 
  226, 128,  73, 160,  54, 248,  22, 194, 142, 209, 164,  92, 123, 121, 227,  98, 
  201, 233, 150, 206, 238, 212,  76,  95, 117, 248,  39,   2, 154, 129, 171,  32, 
  127,  88,  32,  58, 152,  19,  42, 141, 223,  37, 152,  27,  42, 139,  95,  39, 
  120,  26, 162, 139,  57, 167,  82, 250, 189, 131,  49, 161, 212, 120,  95,  98, 
  184,  41, 178, 158, 245, 168,  71,  62, 178, 144, 117, 172,  39,  61, 218, 145, 
  155,  44, 107,  93, 239, 121, 140,  34, 229, 217, 139,  26, 231,  75,  10, 183, 
   71,  54, 178, 150, 245, 174, 199,  60,  82, 145, 253, 172,  65, 189, 240, 113, 
  132,  36,  99,  91, 105, 251, 110, 195, 108,  81, 237, 252,  77, 129, 245, 160, 
   71,  56,  50, 146, 149, 173, 175,  61, 188,  17, 177, 204, 116,  85, 231, 127, 
   10, 160,   7,  56,   2, 146, 129, 173, 160, 125, 184,  33, 178, 152, 117, 170, 
  167,  63,  58, 144,  19,  44,  13, 221, 197, 153, 147,  42, 237, 223,  13, 152, 
    5, 170, 131,  63,  33, 208,  24,  92,  10, 185, 199,  50, 210, 149, 157, 175, 
   41, 188,  30, 241, 200,  68,  86, 179, 126, 245, 224,  71,   8,  50, 134, 149, 
  162, 239,  57, 140,  18, 229, 205, 139,  21, 167,  79,  58, 180,  19,  55,  77, 
  214, 181, 158, 247,  40,  70, 158, 178, 232, 117, 142, 167,  36, 122, 155,  99, 
   43, 105, 223, 110, 216,  44,  90, 157, 251,  41, 131,  94, 225, 248,  72,  66, 
  182, 177, 182, 244, 118, 199, 102, 210, 170, 221, 191,  25, 176,  10, 244,   7, 
    7,  66, 130, 177, 161, 180, 120, 119,  98, 166, 169, 186, 254, 243,   0,  69, 
  192,  51,  16,  21, 204,  15,  21, 196,  15,  19,  68,  13, 243,  69, 133, 243, 
   35,   5, 217, 195,  26, 209, 203,  28,  87,  73, 254, 182, 192, 118, 208,  38, 
  220,  26, 217, 203,  26, 215,  75,  30, 183,  72, 118, 182, 166, 246, 250, 198, 
  195,  18, 209, 205, 156,  85, 169, 255,  62, 192,  16,  80,  12,  60,   5, 209, 
  195,  28,  81, 201, 252,  86, 193, 254, 208,  64,  92,  48,  57, 212,  18, 223, 
   77, 152,  53, 170, 151,  63,  46, 144,  28, 108,   9, 237, 198, 205, 146, 213, 
  173, 159,  61, 168,  17, 190, 140, 112, 101, 228,  43,  11,  95,  71, 120,  50, 
  162, 149, 185, 175,  50, 252,  21, 129, 207,  32,  84,  24,  63,  74, 144,  55, 
   44,  22, 157, 206, 233, 148,  78, 239, 116,  76,  39, 117, 218, 167,  27,  58, 
  139,  83,  39, 125, 218, 161, 155,  56, 107,  82, 175, 125, 188,  33, 177, 216, 
  116,  90, 167, 123,  58, 163,  83,  57, 253, 210, 193, 157, 144, 105, 172,  46, 
  253, 220,  65, 153, 240, 106, 196,  47,  19,  92,  13, 249, 197, 130, 211,  33, 
  157, 216, 105, 154, 174, 235,  60,  79,  81, 244,  60,  71,  81, 242, 188,  69, 
  177, 243,  52,  69, 215, 115,  30, 165, 200, 123,  22, 163,  78, 249, 244,  66, 
  199, 113, 146, 164, 109, 187, 109, 179, 109, 181, 237, 183,  13, 182, 133, 182, 
  227,  54, 201, 214, 214, 222, 222, 216,  88,  90, 186, 187,  51,  51, 255,  63 )

random_mask_vec8 = numpy.array(random_mask_tuple, numpy.uint8)

def whiten(s, o):
    sa = numpy.fromstring(s, numpy.uint8)
    z = sa ^ random_mask_vec8[o:len(sa)+o]
    return z.tostring()

def dewhiten(s, o):
    return whiten(s, o)        # self inverse

class top_block(gr.top_block):

	def __init__(self):
		gr.top_block.__init__(self, "Top Block")

		##################################################
		# Variables
		##################################################
		self.samples_per_symbol = samples_per_symbol = 2
		self.samp_rate = samp_rate = 10000

		##################################################
		# Blocks
		##################################################
		self.gr_file_source_0 = gr.file_source(gr.sizeof_char*1,
						       "preencoder1", False)
		self.gr_file_sink_0 = gr.file_sink(gr.sizeof_char*1, "sink2.png")
		self.gr_file_sink_0.set_unbuffered(True)
                self.gr_file_sink_1 = gr.file_sink(gr.sizeof_char*1, "premod2")
                self.gr_file_sink_1.set_unbuffered(True)
                self.gr_file_sink_2 = gr.file_sink(gr.sizeof_char*1, "postmod2")
                self.gr_file_sink_2.set_unbuffered(True)
                self.gr_file_sink_3 = gr.file_sink(gr.sizeof_gr_complex*1, "midmod2")
                self.gr_file_sink_3.set_unbuffered(True)

		self.digital_dxpsk_mod_0 = generic_mod(
			constellation=digital_swig.constellation_bpsk(),
			samples_per_symbol=samples_per_symbol,
			differential=True,
			excess_bw=0.35,
			gray_coded=True,
			verbose=False,
			log=False)
		#self.digital_dxpsk_mod_0 = digital.dbpsk_mod(
		#	samples_per_symbol=samples_per_symbol,
		#	excess_bw=0.35,
		#	gray_coded=True,
		#	verbose=False,
		#	log=False)
			
		self.digital_dxpsk_demod_1 = generic_demod(
			constellation=digital_swig.constellation_bpsk(),
			samples_per_symbol=samples_per_symbol,
			differential=True,
			excess_bw=0.35,
			gray_coded=True,
			freq_bw=6.28/100.0,
			phase_bw=6.28/100.0,
			timing_bw=6.28/100.0,
			verbose=False,
			log=False
		)
		#self.digital_dxpsk_demod_0 = digital.dbpsk_demod(
		#	samples_per_symbol=samples_per_symbol,
		#	excess_bw=0.35,
		#	freq_bw=6.28/100.0,
		#	phase_bw=6.28/100.0,
		#	timing_bw=6.28/100.0,
		#	gray_coded=True,
		#	verbose=False,
		#	log=False
		#)

		self.packet_encoder_1 = packet_mod_b(packet_encoder(
				samples_per_symbol=samples_per_symbol,
				bits_per_symbol=1,
				access_code="",
				pad_for_usrp=False,
			),
			payload_length=512,
		)
		self.packet_decoder_1 = packet_demod_b(packet_decoder(
				access_code="",
				threshold=-1,
				callback=lambda ok, payload: self.packet_decoder_1.recv_pkt(ok, payload),
			),
		)

		self.test_block_1 = test_block()

                self.throttle1 = gr.throttle(gr.sizeof_char*1, 10000)

		##################################################
		# Connections
		##################################################

		self.connect((self.gr_file_source_0, 0), (self.packet_encoder_1, 0))
                #self.connect((self.gr_file_source_0, 0), (self.digital_dxpsk_mod_0, 0))
		#self.connect((self.gr_file_source_0, 0), (self.test_block_1, 0))
		#self.connect((self.test_block_1, 0), (self.blks2_packet_encoder_1, 0))
		self.connect((self.packet_encoder_1, 0), (self.digital_dxpsk_mod_0, 0))
                #self.connect((self.packet_encoder_1, 0), (self.throttle1, 0))
                #self.connect((self.throttle1, 0), (self.digital_dxpsk_mod_0, 0))
                self.connect((self.packet_encoder_1, 0), (self.gr_file_sink_1, 0))
		#self.connect((self.packet_encoder_1, 0), (self.test_block_1, 0))
		#self.connect((self.test_block_1, 0), (self.digital_dxpsk_mod_0, 0))
		self.connect((self.digital_dxpsk_mod_0, 0), (self.digital_dxpsk_demod_1, 0))
                self.connect((self.digital_dxpsk_mod_0, 0), (self.gr_file_sink_3, 0))
		self.connect((self.digital_dxpsk_demod_1, 0), (self.packet_decoder_1, 0))
		#self.connect((self.digital_dxpsk_demod_1, 0), (self.test_block_1, 0))
                #self.connect((self.digital_dxpsk_demod_1, 0), (self.gr_file_sink_0, 0))
                self.connect((self.digital_dxpsk_demod_1, 0), (self.gr_file_sink_2, 0))
		#self.connect((self.test_block_1, 0), (self.packet_decoder_1, 0))
		#self.connect((self.test_block_1, 0), (self.gr_file_sink_0, 0))
		self.connect((self.packet_decoder_1, 0), (self.gr_file_sink_0, 0))

	def get_samples_per_symbol(self):
		return self.samples_per_symbol

	def set_samples_per_symbol(self, samples_per_symbol):
		self.samples_per_symbol = samples_per_symbol

	def get_samp_rate(self):
		return self.samp_rate

	def set_samp_rate(self, samp_rate):
		self.samp_rate = samp_rate

if __name__ == '__main__':
	parser = OptionParser(option_class=eng_option, usage="%prog: [options]")
	(options, args) = parser.parse_args()
	tb = top_block()
	tb.run()
