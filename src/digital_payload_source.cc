#include <cstdio>

#include "digital_payload_source.h"

#include <gr_io_signature.h>

#include "packet_utils.h"

#define VERBOSE 0

GR_CORE_API digital_payload_source_sptr
digital_make_payload_source(size_t itemsize, gr_msg_queue_sptr msgq,
			    int whitener_offset, bool dewhitening)
{
  return gnuradio::get_initial_sptr(new digital_payload_source(itemsize, msgq,
							       whitener_offset,
							       dewhitening));
}

digital_payload_source::digital_payload_source(size_t itemsize,
					       gr_msg_queue_sptr msgq,
					       int whitener_offset,
					       bool dewhitening):
  gr_sync_block("digital_payload_source",
		gr_make_io_signature(0, 0, 0),
		gr_make_io_signature(1, 1, itemsize)),
  d_itemsize(itemsize), d_msgq(msgq), d_whitener_offset(whitener_offset),
  d_dewhitening(dewhitening)
{
  /* NOP */
}

bool toast = true, toast2 = true;

int digital_payload_source::work(int noutput_items,
				 gr_vector_const_void_star &input_items,
				 gr_vector_void_star &output_items)
{
  char *out = (char *) output_items[0];
  int nn = 0;

  if (VERBOSE)
    fprintf(stderr, "digital_payload_source.work: noutput_items=%d\n", noutput_items);

  // Consume messages and generate packets until nn == noutput_items
  while (nn < noutput_items) {
    if (d_payload_buffer.length() > 0) {
      if (VERBOSE) {
	fprintf(stderr, "payloadsrc: payload_buffer.length=%d, nn=%d, noutput_items=%d\n", d_payload_buffer.length(), nn, noutput_items);
	fprintf(stderr, "payloadsrc: ");
	for (int i = 0; i < 30; i++)
	  fprintf(stderr, "%02X", (unsigned char) d_payload_buffer[i]);
	fprintf(stderr, "\n");
	toast2 = false;
      }

      // Consume whatever we can from the current message
      int mm = std::min(noutput_items - nn, (int) d_payload_buffer.length());
      std::memcpy(out + nn * d_itemsize, d_payload_buffer.data(), mm);
      // Erase what we just copied
      d_payload_buffer.erase(0, mm);
      nn += mm;
    } else {
      // No more messages in queue, return what we've got
      if (d_msgq->empty_p() && nn > 0) {
	break;
      }

      if (VERBOSE)
	fprintf(stderr, "digital_payload_source.work: waiting for message\n");

      // block, waiting for a message
      gr_message_sptr d_msg = d_msgq->delete_head(); 

      if (VERBOSE)
	fprintf(stderr, "digital_payload_source.work: message received\n");

      std::string payload;
      bool ok;
      unmake_packet(payload, ok, (const char*) d_msg->msg(), d_msg->length(),
		    d_whitener_offset, d_dewhitening);
      if (toast) {
	fprintf(stderr, "payloadsrc: ok=%d\n", ok);
	fprintf(stderr, "payloadsrc: ");
	for (int i = 0; i < 30; i++) {
	  fprintf(stderr, "%02X", (unsigned char) payload[i]);
	}
	fprintf(stderr, "\n");
	toast = false;
      }

      if (ok) {
	d_payload_buffer.append(payload);
	
	if (VERBOSE)
	  fprintf(stderr, "digital_payload_source.work: packet OK\n");
      } else {
	if (VERBOSE)
	  fprintf(stderr, "digital_payload_source.work: packet BAD ***\n");
      }
    }
  }

  if (VERBOSE)
    fprintf(stderr, "digital_payload_source.work: returning nn=%d\n", nn);

  return nn;
}
