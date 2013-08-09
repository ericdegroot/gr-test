#include <boost/array.hpp>

#include "digital_generic_demod.h"

#include <gr_io_signature.h>
#include <gr_firdes.h>

#include <gr_file_sink.h>

#define DEBUG 1

struct code_sort_pred
{
  inline bool operator() (const boost::array<int, 2>& a,
			  const boost::array<int, 2>& b)
  {
    return (a[1] == b[1] && a[0] < b[0]) || (a[1] < b[1]);
  }
};

// This follows the pattern of decorate-sort-undecorate used by the python
// map_codes.invert_code(code) method and may not be the best c++ solution...
std::vector<int> invert_code(std::vector<unsigned int> code) {
  std::vector<boost::array<int, 2> > decorated_code(code.size());
  for (int i = 0; i != code.size(); i++) {
    decorated_code[i][0] = i;
    decorated_code[i][1] = code[i];
  }

  std::sort(decorated_code.begin(), decorated_code.end(), code_sort_pred());

  std::vector<int> inverted_code(decorated_code.size());
  for (int i = 0; i != decorated_code.size(); i++)
    inverted_code[i] = decorated_code[i][1];

  return inverted_code;
}

DIGITAL_API digital_generic_demod_sptr
digital_make_generic_demod(digital_constellation_bpsk_sptr constellation,
			   float samples_per_symbol, bool differential,
			   float excess_bw, bool gray_coded, float freq_bw,
			   float timing_bw, float phase_bw)
{
  return gnuradio::get_initial_sptr(new digital_generic_demod(constellation,
							      samples_per_symbol,
							      differential,
							      excess_bw,
							      gray_coded,
							      freq_bw,
							      timing_bw,
							      phase_bw));
}

digital_generic_demod::digital_generic_demod(digital_constellation_bpsk_sptr
					     constellation,
					     float samples_per_symbol,
					     bool differential, float excess_bw,
					     bool gray_coded, float freq_bw,
					     float timing_bw, float phase_bw):
  gr_hier_block2("digital_generic_mod",
		 gr_make_io_signature(1, 1, sizeof(gr_complex)),
		 gr_make_io_signature(1, 1, sizeof(char))),
  d_constellation(constellation), d_samples_per_symbol(samples_per_symbol),
  d_differential(differential), d_excess_bw(excess_bw), d_gray_coded(gray_coded),
  d_freq_bw(freq_bw), d_timing_bw(timing_bw), d_phase_bw(phase_bw)
{
  int bits_per_symbol = d_constellation->bits_per_symbol();
  int arity = pow(2, bits_per_symbol);

  unsigned int nfilts = 32;
  double ntaps = nfilts * 11 * d_samples_per_symbol; // make nfilts filters of
                                                     // ntaps each

  float init_phase = nfilts / 2;
  float timing_max_dev = 1.5;

  // automatic gain control
  d_agc = gr_make_agc2_cc(0.6e-1, 1e-3, 1, 1, 100);

  // frequency correction
  int fll_ntaps = 55;
  d_freq_recov = digital_make_fll_band_edge_cc(d_samples_per_symbol, d_excess_bw,
						 fll_ntaps, d_freq_bw);

  // symbol timing recovery with RRC data filter
  static std::vector<float> taps =
    gr_firdes::root_raised_cosine(nfilts, nfilts * d_samples_per_symbol, 1.0,
				    d_excess_bw, ntaps);
  d_time_recov = digital_make_pfb_clock_sync_ccf(d_samples_per_symbol,
						 d_timing_bw, taps, nfilts,
						 init_phase, timing_max_dev);

  float fmin = -0.25;
  float fmax = 0.25;
  d_receiver = digital_make_constellation_receiver_cb(d_constellation, d_phase_bw,
						      fmin, fmax);

  // do differential decoding based on phase change of symbols
  d_diffdec = digital_make_diff_decoder_bb(arity);

  std::vector<unsigned int> pre_diff_code_uint = constellation->pre_diff_code();
  std::vector<int> inverted_code = invert_code(pre_diff_code_uint);
  d_symbol_mapper = digital_make_map_bb(inverted_code);

  d_unpack = gr_make_unpack_k_bits_bb(bits_per_symbol);

  // Connect
  connect(self(), 0, d_agc, 0);
  connect(d_agc, 0, d_freq_recov, 0);
  connect(d_freq_recov, 0, d_time_recov, 0);
  connect(d_time_recov, 0, d_receiver, 0);
  connect(d_receiver, 0, d_diffdec, 0);

  if (d_constellation->apply_pre_diff_code()) {
    connect(d_diffdec, 0, d_symbol_mapper, 0);
    connect(d_symbol_mapper, 0, d_unpack, 0);
  } else {
    connect(d_diffdec, 0, d_unpack, 0);
  }

  connect(d_unpack, 0, self(), 0);

  if (DEBUG) {
    //gr_complex_dump_cb_sptr complex_dump0 = gr_make_complex_dump_cb(4);
    //gr_file_sink_sptr sink0 = gr_make_file_sink(sizeof(unsigned char), "agc1");
    //sink0->set_unbuffered(true);
    //gr_complex_dump_cb_sptr complex_dump1 = gr_make_complex_dump_cb(4);
    //gr_file_sink_sptr sink1 = gr_make_file_sink(sizeof(unsigned char), "freq_recov1");
    //sink1->set_unbuffered(true);
    //gr_complex_dump_cb_sptr complex_dump2 = gr_make_complex_dump_cb(4);
    //gr_file_sink_sptr sink2 = gr_make_file_sink(sizeof(unsigned char), "time_recov1");
    //sink2->set_unbuffered(true);
    gr_file_sink_sptr sink3 = gr_make_file_sink(sizeof(char), "log/receiver1");
    sink3->set_unbuffered(true);
    gr_file_sink_sptr sink4 = gr_make_file_sink(sizeof(char), "log/diffdec1");
    sink4->set_unbuffered(true);
    gr_file_sink_sptr sink5 = gr_make_file_sink(sizeof(char), "log/demod_symbol_mapper1");
    sink5->set_unbuffered(true);
    gr_file_sink_sptr sink6 = gr_make_file_sink(sizeof(char), "log/unpack1");
    sink6->set_unbuffered(true);
    gr_file_sink_sptr sink7 = gr_make_file_sink(sizeof(gr_complex), "log/agc1bin");
    sink7->set_unbuffered(true);
    gr_file_sink_sptr sink8 = gr_make_file_sink(sizeof(gr_complex), "log/freq_recov1bin");
    sink8->set_unbuffered(true);
    gr_file_sink_sptr sink9 = gr_make_file_sink(sizeof(gr_complex), "log/time_recov1bin");
    sink9->set_unbuffered(true);

    connect(d_agc, 0, sink7, 0);
    connect(d_freq_recov, 0, sink8, 0);
    connect(d_time_recov, 0, sink9, 0);

    //connect(d_agc, 0, complex_dump0, 0);
    //connect(complex_dump0, 0, sink0, 0);
    //connect(d_freq_recov, 0, complex_dump1, 0);    
    //connect(complex_dump1, 0, sink1, 0);
    //connect(d_time_recov, 0, complex_dump2, 0);
    //connect(complex_dump2, 0, sink2, 0);

    connect(d_receiver, 0, sink3, 0);
    connect(d_diffdec, 0, sink4, 0);

    if (d_constellation->apply_pre_diff_code())
      connect(d_symbol_mapper, 0, sink5, 0);

    connect(d_unpack, 0, sink6, 0);
  }
}
