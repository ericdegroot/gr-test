#include <cstdio>

#include "digital_packet_encoder.h"
#include "packet_utils.h"

#include <gr_io_signature.h>

DIGITAL_API digital_packet_encoder_sptr
digital_make_packet_encoder(size_t payload_len, int samples_per_symbol,
                            int bits_per_symbol, bool pad_for_usrp,
                            int whitener_offset, bool whitening)
{
  return gnuradio::get_initial_sptr(new digital_packet_encoder(payload_len,
                                                               samples_per_symbol,
                                                               bits_per_symbol,
                                                               pad_for_usrp,
                                                               whitener_offset,
                                                               whitening));
}

digital_packet_encoder::digital_packet_encoder(size_t payload_len,
                                               int samples_per_symbol,
                                               int bits_per_symbol,
                                               bool pad_for_usrp,
                                               int whitener_offset,
                                               bool whitening):
  gr_block("digital_packet_encoder",
           gr_make_io_signature(1, 1, sizeof(char)),
           gr_make_io_signature(1, 1, sizeof(char))),
  d_payload_len(payload_len), d_samples_per_symbol(samples_per_symbol),
  d_bits_per_symbol(bits_per_symbol), d_pad_for_usrp(pad_for_usrp),
  d_whitener_offset(whitener_offset), d_whitening(whitening)
{
  /* NOP */
}

bool toast6 = true;

int digital_packet_encoder::general_work(int noutput_items,
                                         gr_vector_int &ninput_items,
                                         gr_vector_const_void_star &input_items,
                                         gr_vector_void_star &output_items)
{
  int n_in = ninput_items[0];
  const char *in = (const char *) input_items[0];
  char *out = (char *) output_items[0];

  int packet_size = packed_preamble_length + packed_default_access_code_length +
    4 + d_payload_len + 4 + 1;

  int consumed = 0, produced = 0;
  while (produced + packet_size < noutput_items) {
    std::string packet;
    make_packet(packet, in + consumed, d_payload_len, d_samples_per_symbol,
                d_bits_per_symbol, default_access_code, d_pad_for_usrp,
                d_whitener_offset, d_whitening);

    if (toast6) {
      fprintf(stderr, "pos-makpak: ");
      for (int i = 14; i < 44; i++)
        fprintf(stderr, "%02X", (unsigned char) packet[i]);
      fprintf(stderr, "\n");
      toast6 = false;
    }

    // copy produced packet to output buffer
    std::memcpy(out, packet.data(), packet_size);

    out += packet_size;
    consumed += d_payload_len;
    produced += packet_size;
  }

  consume_each(consumed);

  return produced;
}

void digital_packet_encoder::forecast(int noutput_items,
                                      gr_vector_int &ninput_items_required)
{
  int packet_size = packed_preamble_length + packed_default_access_code_length +
    4 + d_payload_len + 4 + 1;

  // we always require enough input for at least 1 payload
  // int num_packets = std::max(1, noutput_items / packet_size);
  int num_packets = noutput_items / packet_size;
  ninput_items_required[0] = num_packets * d_payload_len;
}
