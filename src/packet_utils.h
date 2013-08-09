#include <string>
#include <arpa/inet.h>

extern const std::string default_access_code;
extern const std::string packed_default_access_code;
extern const size_t packed_default_access_code_length;

extern const std::string preamble;
extern const std::string packed_preamble;
extern const size_t packed_preamble_length;

void conv_1_0_string_to_packed_binary_string(std::string&, bool&,
					     const char*, size_t);
void conv_packed_binary_string_to_1_0_string(std::string&, const char*,
					     size_t);

void make_header(std::string& header, uint16_t payload_len);
int _npadding_bytes(int pkt_byte_len, int samples_per_symbol, int bits_per_symbol);
void make_packet(std::string& packet, const char* payload, size_t payload_len,
		 int samples_per_symbol, int bits_per_symbol,
		 const std::string access_code, bool pad_for_usrp,
		 int whitener_offset, bool whitening);

void unmake_packet(std::string&, bool&, const char*, size_t, int, bool);
std::string whiten(const char*, size_t, int, bool);
extern const unsigned char random_mask_tuple[];
