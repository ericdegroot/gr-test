#include <gr_core_api.h>
#include <gr_block.h>

class gr_complex_dump_cb;
typedef boost::shared_ptr<gr_complex_dump_cb> gr_complex_dump_cb_sptr;
GR_CORE_API gr_complex_dump_cb_sptr gr_make_complex_dump_cb(int precision);

class GR_CORE_API gr_complex_dump_cb : public gr_block
{
private:
  friend GR_CORE_API gr_complex_dump_cb_sptr gr_make_complex_dump_cb(int precision);

  int d_precision;
  char d_format[63];
  int d_msg_len;

  gr_complex_dump_cb(int precision);

public:
  int general_work(int noutput_items, gr_vector_int &ninput_items,
		   gr_vector_const_void_star &input_items,
		   gr_vector_void_star &output_items);

  void forecast(int noutput_items, gr_vector_int &ninput_items_required);
};
