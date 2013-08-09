#include <cstdio>

#include "gr_complex_dump_cb.h"

GR_CORE_API gr_complex_dump_cb_sptr gr_make_complex_dump_cb(int precision)
{
  return gnuradio::get_initial_sptr(new gr_complex_dump_cb(precision));
}

gr_complex_dump_cb::gr_complex_dump_cb(int precision):
  gr_block("gr_complex_dump_cb",
	   gr_make_io_signature(1, 1, sizeof(gr_complex)),
	   gr_make_io_signature(1, 1, sizeof(char))),
  d_precision(precision)
{
  sprintf(d_format, "%%c%%.0%df%%ci%%.0%df\n", d_precision, d_precision);
  d_msg_len = 2 * d_precision + 8;
}

int gr_complex_dump_cb::general_work(int noutput_items, gr_vector_int &ninput_items,
				     gr_vector_const_void_star &input_items,
				     gr_vector_void_star &output_items)
{
  int n_in = ninput_items[0];
  const gr_complex *in = (const gr_complex *) input_items[0];
  char *out = (char *) output_items[0];

  int consumed = 0, produced = 0;
  while (consumed + 1 <= n_in && produced + d_msg_len <= noutput_items) {
    char real_sign = in[consumed].real() < 0 ? '-' : '+';
    char imag_sign = in[consumed].imag() < 0 ? '-' : '+';
    sprintf(out + produced, d_format, real_sign, fabs(in[consumed].real()),
	    imag_sign, fabs(in[consumed].imag()));

    consumed++;
    produced += d_msg_len;
  }

  consume_each(consumed);

  return produced;
}

void gr_complex_dump_cb::forecast(int noutput_items,
				  gr_vector_int &ninput_items_required)
{
  ninput_items_required[0] = noutput_items / d_msg_len;
}
