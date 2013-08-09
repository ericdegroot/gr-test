#include <cstdio>
#include <string>

#include "packet_utils.h"

int main(int argc, char **argv)
{
  /*
  bool padded;
  std::string ps;

  conv_1_0_string_to_packed_binary_string(ps, padded, "10101111", 8);
  printf("length=%d, padded=%d, ps=%s\n", ps.length(), (int) padded, ps.c_str());

  ps.clear();

  conv_1_0_string_to_packed_binary_string(ps, padded, "1010111", 7);
  printf("length=%d, padded=%d, ps=%s\n", ps.length(), (int) padded, ps.c_str());

  ps.clear();

  conv_1_0_string_to_packed_binary_string(ps, padded, "0101111", 7);
  printf("length=%d, padded=%d, ps=%s\n", ps.length(), (int) padded, ps.c_str());

  ps.clear();

  conv_1_0_string_to_packed_binary_string(ps, padded, "1111010110101111", 16);
  printf("length=%d, padded=%d, ps=%s\n", ps.length(), (int) padded, ps.c_str());

  ps.clear();

  conv_1_0_string_to_packed_binary_string(ps, padded, "111010110101111", 15);
  printf("length=%d, padded=%d, ps=%s\n", ps.length(), (int) padded, ps.c_str());

  ps.clear();

  conv_1_0_string_to_packed_binary_string(ps, padded, "111101011010111", 15);
  printf("length=%d, padded=%d, ps=%s\n", ps.length(), (int) padded, ps.c_str());

  std::string bs;

  conv_packed_binary_string_to_1_0_string(bs, "\xAF", 1);
  printf("length=%d, bs=%s\n", bs.length(), bs.c_str());

  bs.clear();

  conv_packed_binary_string_to_1_0_string(bs, "\xAC\xDD\xA4\xE2\xF2\x8C\x20\xFC", 8);
  printf("length=%d, bs=%s\n", bs.length(), bs.c_str());

  bs.clear();

  conv_packed_binary_string_to_1_0_string(bs, "\xA4\xF2", 2);
  printf("length=%d, bs=%s\n", bs.length(), bs.c_str());  

  std::string h;

  make_header(h, 512);
  printf("h=%s\n", h.c_str());

  std::string packet;
  const char* payload_in = "toast";
  make_packet(packet, payload_in, 5, default_access_code);

  size_t header_len = packed_preamble_length + packed_default_access_code_length + 4;
  size_t payload_out_len = packet.length() - (header_len + 1);
  std::string payload_crc_out = packet.substr(header_len, payload_out_len); 
  printf("packet=%s, packet.length=%d, payload_crc_out=%s\n", packet.c_str(), packet.length(), payload_crc_out.c_str());

  std::string payload_out;
  bool ok;
  unmake_packet(payload_out, ok, payload_crc_out.data(), payload_crc_out.length());

  printf("payload=%s, ok=%d\n", payload_out.c_str(), ok);
  */

  for (int i = 0; i < 512; i++) {
    printf("%d %d\n", i, _npadding_bytes(512 + i, 2, 1));
  }

  return 0;
}
