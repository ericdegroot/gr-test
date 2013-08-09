#include "gr_framer_sink_1.h"
#include <gr_io_signature.h>
#include <cstdio>
#include <stdexcept>
#include <string.h>

#define VERBOSE 1

inline void
gr_framer_sink_1::enter_search()
{
  if (VERBOSE)
    fprintf(stderr, "\t@ enter_search\n");

  d_state = STATE_SYNC_SEARCH;
}

inline void
gr_framer_sink_1::enter_have_sync()
{
  if (VERBOSE)
    fprintf(stderr, "\t@ enter_have_sync\n");

  d_state = STATE_HAVE_SYNC;
  d_header = 0;
  d_headerbitlen_cnt = 0;
}

inline void
gr_framer_sink_1::enter_have_header(int payload_len, int whitener_offset)
{
  if (VERBOSE)
    fprintf(stderr, "\t@ enter_have_header (payload_len = %d) (offset = %d)\n", payload_len, whitener_offset);

  d_state = STATE_HAVE_HEADER;
  d_packetlen = payload_len;
  d_packet_whitener_offset = whitener_offset;
  d_packetlen_cnt = 0;
  d_packet_byte = 0;
  d_packet_byte_index = 0;
}

gr_framer_sink_1_sptr
gr_make_framer_sink_1(gr_msg_queue_sptr target_queue)
{
  return gnuradio::get_initial_sptr(new gr_framer_sink_1(target_queue));
}


gr_framer_sink_1::gr_framer_sink_1(gr_msg_queue_sptr target_queue)
  : gr_sync_block ("framer_sink_1",
		   gr_make_io_signature (1, 1, sizeof(unsigned char)),
		   gr_make_io_signature (0, 0, 0)),
    d_target_queue(target_queue)
{
  enter_search();
}

gr_framer_sink_1::~gr_framer_sink_1 ()
{
}

int
gr_framer_sink_1::work (int noutput_items,
			gr_vector_const_void_star &input_items,
			gr_vector_void_star &output_items)
{
  const unsigned char *in = (const unsigned char *) input_items[0];
  int count=0;

  if (VERBOSE)
    fprintf(stderr,"gr_framer_sink_1.work: noutput_items=%d\n", noutput_items);

  while (count < noutput_items){
    switch(d_state) {

    case STATE_SYNC_SEARCH:    // Look for flag indicating beginning of pkt
      if (VERBOSE)
	fprintf(stderr,"\tSYNC Search, noutput=%d\n", noutput_items);

      while (count < noutput_items) {
	if (in[count] & 0x2){  // Found it, set up for header decode
	  enter_have_sync();
	  break;
	}
	count++;
      }
      if (VERBOSE && count >= noutput_items)
	fprintf(stderr,"\tFlag not found this time\n");
      break;

    case STATE_HAVE_SYNC:
      if (VERBOSE)
	fprintf(stderr,"\tHeader Search bitcnt=%d, header=0x%08x\n",
		d_headerbitlen_cnt, d_header);

      while (count < noutput_items) {	// Shift bits one at a time into header
	d_header = (d_header << 1) | (in[count++] & 0x1);
	if (++d_headerbitlen_cnt == HEADERBITLEN) {

	  if (VERBOSE)
	    fprintf(stderr, "\t\tgot header: d_header=0x%08x, count=%d\n", d_header, count);

	  // we have a full header, check to see if it has been received properly
	  if (header_ok()){
	    int payload_len;
	    int whitener_offset;
	    header_payload(&payload_len, &whitener_offset);
	    enter_have_header(payload_len, whitener_offset);

	    if (d_packetlen == 0){	    // check for zero-length payload
	      // build a zero-length message
	      // NOTE: passing header field as arg1 is not scalable
	      gr_message_sptr msg =
		gr_make_message(0, d_packet_whitener_offset, 0, 0);

	      d_target_queue->insert_tail(msg);		// send it
	      msg.reset();  				// free it up

	      enter_search();
	    }
	  }
	  else
	    enter_search();				// bad header
	  break;					// we're in a new state
	}
      }
      break;

    case STATE_HAVE_HEADER:
      if (VERBOSE)
	fprintf(stderr,"\tPacket Build d_packetlen_cnt=%d, count=%d\n", d_packetlen_cnt, count);

      while (count < noutput_items) {   // shift bits into bytes of packet one at a time
	d_packet_byte = (d_packet_byte << 1) | (in[count++] & 0x1);
	if (d_packet_byte_index++ == 7) {	  	// byte is full so move to next byte
	  d_packet[d_packetlen_cnt++] = d_packet_byte;
	  d_packet_byte_index = 0;

	  if (d_packetlen_cnt == d_packetlen){		// packet is filled

	    // build a message
	    // NOTE: passing header field as arg1 is not scalable
	    gr_message_sptr msg =
	      gr_make_message(0, d_packet_whitener_offset, 0, d_packetlen_cnt);
	    memcpy(msg->msg(), d_packet, d_packetlen_cnt);

	    fprintf(stderr,"gr_framer_sink1.work: INSERTING MESSAGE\n");

	    d_target_queue->insert_tail(msg);		// send it
	    msg.reset();  				// free it up

	    enter_search();
	    break;
	  }
	}
      }
      fprintf(stderr, "\tLeaving %d\n", d_packetlen_cnt);
      break;

    default:
      assert(0);

    } // switch

  }   // while

  return noutput_items;
}
