#include <digital_api.h>
#include <gr_core_api.h>
#include <gr_block.h>

class digital_packet_encoder;
typedef boost::shared_ptr<digital_packet_encoder> digital_packet_encoder_sptr;
DIGITAL_API digital_packet_encoder_sptr
digital_make_packet_encoder(size_t payload_len, int samples_per_symbol,
			    int bits_per_symbol, bool pad_for_usrp,
			    int whitener_offset, bool whitening);

class DIGITAL_API digital_packet_encoder : public gr_block
{
private:
  friend DIGITAL_API digital_packet_encoder_sptr
    digital_make_packet_encoder(size_t payload_len, int samples_per_symbol,
				int bits_per_symbol, bool pad_for_usrp,
				int whitener_offset, bool whitening);

  size_t d_payload_len;
  int d_samples_per_symbol;
  int d_bits_per_symbol;
  bool d_pad_for_usrp;
  int d_whitener_offset;
  bool d_whitening;
  long d_total_consumed;
  long d_total_produced;

  digital_packet_encoder(size_t payload_len, int samples_per_symbol,
			 int bits_per_symbol, bool pad_for_usrp,
			 int whitener_offset, bool whitening);

public:
  int general_work(int noutput_items, gr_vector_int &ninput_items,
		   gr_vector_const_void_star &input_items,
		   gr_vector_void_star &output_items);

  void forecast(int noutput_items, gr_vector_int &ninput_items_required);
};
