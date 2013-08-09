#include <gr_core_api.h>
#include <gr_sync_block.h>
#include <gr_message.h>
#include <gr_msg_queue.h>

class gr_message_source;
typedef boost::shared_ptr<gr_message_source> gr_message_source_sptr;

GR_CORE_API gr_message_source_sptr gr_make_message_source (size_t itemsize, int msgq_limit=0);
GR_CORE_API gr_message_source_sptr gr_make_message_source (size_t itemsize, gr_msg_queue_sptr msgq);

/*!
 * \brief Turn received messages into a stream
 * \ingroup source_blk
 */
class GR_CORE_API gr_message_source : public gr_sync_block
{
 private:
  size_t	 	d_itemsize;
  gr_msg_queue_sptr	d_msgq;
  gr_message_sptr	d_msg;
  unsigned		d_msg_offset;
  bool			d_eof;

  friend GR_CORE_API gr_message_source_sptr
  gr_make_message_source(size_t itemsize, int msgq_limit);
  friend GR_CORE_API gr_message_source_sptr
  gr_make_message_source(size_t itemsize, gr_msg_queue_sptr msgq);

 protected:
  gr_message_source (size_t itemsize, int msgq_limit);
  gr_message_source (size_t itemsize, gr_msg_queue_sptr msgq);

 public:
  ~gr_message_source ();

  gr_msg_queue_sptr	msgq() const { return d_msgq; }

  int work (int noutput_items,
	    gr_vector_const_void_star &input_items,
	    gr_vector_void_star &output_items);
};
