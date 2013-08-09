#include "gr_message_source.h"
#include <gr_io_signature.h>
#include <cstdio>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdexcept>
#include <string.h>


// public constructor that returns a shared_ptr

gr_message_source_sptr
gr_make_message_source(size_t itemsize, int msgq_limit)
{
  return gnuradio::get_initial_sptr(new gr_message_source(itemsize, msgq_limit));
}

// public constructor that takes existing message queue
gr_message_source_sptr
gr_make_message_source(size_t itemsize, gr_msg_queue_sptr msgq)
{
  return gnuradio::get_initial_sptr(new gr_message_source(itemsize, msgq));
}

gr_message_source::gr_message_source (size_t itemsize, int msgq_limit)
  : gr_sync_block("message_source",
		  gr_make_io_signature(0, 0, 0),
		  gr_make_io_signature(1, 1, itemsize)),
    d_itemsize(itemsize), d_msgq(gr_make_msg_queue(msgq_limit)), d_msg_offset(0), d_eof(false)
{
}

gr_message_source::gr_message_source (size_t itemsize, gr_msg_queue_sptr msgq)
  : gr_sync_block("message_source",
		  gr_make_io_signature(0, 0, 0),
		  gr_make_io_signature(1, 1, itemsize)),
    d_itemsize(itemsize), d_msgq(msgq), d_msg_offset(0), d_eof(false)
{
}

gr_message_source::~gr_message_source()
{
}

int
gr_message_source::work(int noutput_items,
			gr_vector_const_void_star &input_items,
			gr_vector_void_star &output_items)
{
  char *out = (char *) output_items[0];
  int nn = 0;

  while (nn < noutput_items){
    if (d_msg){
      //
      // Consume whatever we can from the current message
      //
      int mm = std::min(noutput_items - nn, (int)((d_msg->length() - d_msg_offset) / d_itemsize));
      memcpy (out, &(d_msg->msg()[d_msg_offset]), mm * d_itemsize);

      nn += mm;
      out += mm * d_itemsize;
      d_msg_offset += mm * d_itemsize;
      assert(d_msg_offset <= d_msg->length());

      if (d_msg_offset == d_msg->length()){
	if (d_msg->type() == 1)	           // type == 1 sets EOF
	  d_eof = true;
	d_msg.reset();
      }
    }
    else {
      //
      // No current message
      //
      if (d_msgq->empty_p() && nn > 0){    // no more messages in the queue, return what we've got
	break;
      }

      if (d_eof)
	return -1;

      fprintf(stderr,"gr_message_source.work: waiting for message\n");

      d_msg = d_msgq->delete_head();	   // block, waiting for a message
      d_msg_offset = 0;

      fprintf(stderr,"gr_message_source.work: message received\n");

      if ((d_msg->length() % d_itemsize) != 0)
	throw std::runtime_error("msg length is not a multiple of d_itemsize");
    }
  }

  return nn;
}
