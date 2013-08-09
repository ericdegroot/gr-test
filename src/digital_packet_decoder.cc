#include "digital_packet_decoder.h"

#include <gr_io_signature.h>

#include "packet_utils.h"

DIGITAL_API digital_packet_decoder_sptr
digital_make_packet_decoder(size_t payload_len, int whitener_offset,
			    bool dewhitening)
{
  return gnuradio::get_initial_sptr(new digital_packet_decoder(payload_len,
							       whitener_offset,
							       dewhitening));
}

digital_packet_decoder::digital_packet_decoder(size_t payload_len,
					       int whitener_offset,
					       bool dewhitening):
  gr_hier_block2("digital_packet_decoder",
		 gr_make_io_signature(1, 1, sizeof(char)),
		 gr_make_io_signature(1, 1, sizeof(char))),
  d_payload_len(payload_len), d_whitener_offset(whitener_offset),
  d_dewhitening(dewhitening)
{
  msgq = gr_make_msg_queue(DEFAULT_MSGQ_LIMIT);
  correlater = digital_make_correlate_access_code_bb(default_access_code,
						       DEFAULT_THRESHOLD);    
  framer = gr_make_framer_sink_1(msgq);
  msg_source = digital_make_payload_source(sizeof(char), msgq, d_whitener_offset,
					   d_dewhitening);

  connect(self(), 0, correlater, 0);
  connect(correlater, 0, framer, 0);
  connect(msg_source, 0, self(), 0);
}
