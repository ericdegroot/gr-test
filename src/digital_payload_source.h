#include <digital_api.h>
#include <gr_core_api.h>
#include <gr_sync_block.h>
#include <gr_msg_queue.h>

class digital_payload_source;
typedef boost::shared_ptr<digital_payload_source> digital_payload_source_sptr;
DIGITAL_API digital_payload_source_sptr
digital_make_payload_source(size_t itemsize, gr_msg_queue_sptr msgq,
			    int whitener_offset, bool dewhitening);

class GR_CORE_API digital_payload_source : public gr_sync_block
{
private:
  friend GR_CORE_API digital_payload_source_sptr
    digital_make_payload_source(size_t itemsize, gr_msg_queue_sptr msgq,
				int whitener_offset, bool dewhitening);

  size_t d_itemsize;
  gr_msg_queue_sptr d_msgq;
  int d_whitener_offset;
  bool d_dewhitening;
  std::string d_payload_buffer;

  digital_payload_source(size_t itemsize, gr_msg_queue_sptr msgq,
			 int whitener_offset, bool dewhitening);

public:
  int work(int noutput_items, gr_vector_const_void_star &input_items,
	   gr_vector_void_star &output_items);
};
