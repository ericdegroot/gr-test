#include "digital_generic_mod.h"

#include <gr_io_signature.h>

#include <gr_endianness.h>
#include <gr_firdes.h>

#include <gr_file_sink.h>

#define VERBOSE 0
#define DEBUG 1

DIGITAL_API digital_generic_mod_sptr
digital_make_generic_mod(digital_constellation_bpsk_sptr constellation,
			   float samples_per_symbol, bool differential,
			   float excess_bw, bool gray_coded)
{
  return gnuradio::get_initial_sptr(new digital_generic_mod(constellation,
							      samples_per_symbol,
							      differential,
							      excess_bw,
							      gray_coded));
}

digital_generic_mod::digital_generic_mod(digital_constellation_bpsk_sptr
					 constellation,
					 float samples_per_symbol,
					 bool differential, float excess_bw,
					 bool gray_coded):
  gr_hier_block2("digital_generic_mod", gr_make_io_signature(1, 1, sizeof(char)),
		 gr_make_io_signature(1, 1, sizeof(gr_complex))),
  d_constellation(constellation), d_samples_per_symbol(samples_per_symbol),
  d_differential(differential), d_excess_bw(excess_bw), d_gray_coded(gray_coded)
{
  int bits_per_symbol = d_constellation->bits_per_symbol();
  int arity = pow(2, bits_per_symbol);

  // turn bytes into k-bit vectors
  d_bytes2chunks = gr_make_packed_to_unpacked_bb(bits_per_symbol, GR_MSB_FIRST);  

  // convert uint vector to int vector and hope the values all fit...
  std::vector<unsigned int> pre_diff_code_uint = d_constellation->pre_diff_code();
  fprintf(stderr, "pre_diff_code=%d\n", pre_diff_code_uint.size());
  std::vector<int> pre_diff_code_int(pre_diff_code_uint.begin(),
				     pre_diff_code_uint.end());

  d_symbol_mapper = digital_make_map_bb(pre_diff_code_int);
  d_diffenc = digital_make_diff_encoder_bb(arity);
  d_chunks2symbols = digital_make_chunks_to_symbols_bc(d_constellation->points());

  // pulse shaping filter
  unsigned int nfilts = 32;
  int ntaps = nfilts * 11 * samples_per_symbol; // make nfilts filters of
                                                // ntaps each
  static std::vector<float> rrc_taps =
    gr_firdes::root_raised_cosine(nfilts, // gain
				  nfilts, // sampling rate
				  1.0, // symbol rate
				  d_excess_bw, // roll-off factor
				  ntaps);

  fprintf(stderr, "rrc_taps=%d, ntaps=%d, nfilts=%d\n", rrc_taps.size(), ntaps, nfilts);

  /*
  for (int i = 0; i < rrc_taps.size(); i++) {
    fprintf(stderr, "%f\t", rrc_taps[i]);
  }
  fprintf(stderr, "\n");
  */

  d_rrc_filter = gr_make_pfb_arb_resampler_ccf(d_samples_per_symbol, rrc_taps, 32);

  // connect
  connect(self(), 0, d_bytes2chunks, 0);
  connect(d_bytes2chunks, 0, d_symbol_mapper, 0);
  connect(d_symbol_mapper, 0, d_diffenc, 0);
  connect(d_diffenc, 0, d_chunks2symbols, 0);
  connect(d_chunks2symbols, 0, d_rrc_filter, 0);
  connect(d_rrc_filter, 0, self(), 0);

  if (VERBOSE) {
    // fprintf(stderr, "nfilts=%d, d_excess_bw=%.2f, ntaps=%d, samples_per_symbol=%.2f, rrc_taps=%d\n", nfilts, d_excess_bw, ntaps, d_samples_per_symbol, rrc_taps.size());

    //for (int i = 0; i < rrc_taps.size(); i++)
    //  fprintf(stderr, "%.2f", rrc_taps[i]);
    //fprintf(stderr, "\n");
  }

  if (DEBUG) {
    gr_file_sink_sptr sink0 = gr_make_file_sink(sizeof(unsigned char), "log/bytes2chunks1");
    sink0->set_unbuffered(true);
    gr_file_sink_sptr sink1 = gr_make_file_sink(sizeof(unsigned char), "log/symbol_mapper1");
    sink1->set_unbuffered(true);
    gr_file_sink_sptr sink2 = gr_make_file_sink(sizeof(unsigned char), "log/diffenc1");
    sink2->set_unbuffered(true);
    gr_file_sink_sptr sink3 = gr_make_file_sink(sizeof(gr_complex), "log/chunks2symbols1");
    sink3->set_unbuffered(true);
    gr_file_sink_sptr sink4 = gr_make_file_sink(sizeof(gr_complex), "log/rrc_filter1");
    sink4->set_unbuffered(true);

    gr_file_sink_sptr sink5 = gr_make_file_sink(sizeof(unsigned char), "log/generic_mod_in1");
    sink5->set_unbuffered(true);

    connect(self(), 0, sink5, 0);
    connect(d_bytes2chunks, 0, sink0, 0);
    connect(d_symbol_mapper, 0, sink1, 0);
    connect(d_diffenc, 0, sink2, 0);
    connect(d_chunks2symbols, 0, sink3, 0);
    connect(d_rrc_filter, 0, sink4, 0);
  }
}
