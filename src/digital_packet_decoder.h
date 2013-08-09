#include <digital_api.h>
#include <gr_core_api.h>
#include <gr_hier_block2.h>

#include <gr_framer_sink_1.h>
#include <gr_msg_queue.h>
#include <digital_correlate_access_code_bb.h>

#include "digital_payload_source.h"

#define DEFAULT_MSGQ_LIMIT 2
#define DEFAULT_THRESHOLD 12

class digital_packet_decoder;
typedef boost::shared_ptr<digital_packet_decoder> digital_packet_decoder_sptr;
DIGITAL_API digital_packet_decoder_sptr
digital_make_packet_decoder(size_t payload_len, int whitener_offset,
			    bool dewhitening);

class DIGITAL_API digital_packet_decoder : public gr_hier_block2
{
private:
  friend DIGITAL_API digital_packet_decoder_sptr
    digital_make_packet_decoder(size_t payload_len, int whitener_offset,
				bool dewhitening);

  size_t d_payload_len;
  int d_whitener_offset;
  bool d_dewhitening;
  gr_msg_queue_sptr msgq;
  digital_correlate_access_code_bb_sptr correlater;
  gr_framer_sink_1_sptr framer;
  digital_payload_source_sptr msg_source;

  digital_packet_decoder(size_t payload_len, int whitener_offset, bool dewhitening);
};
