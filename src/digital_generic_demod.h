#include <digital_api.h>
#include <gr_core_api.h>
#include <gr_hier_block2.h>

#include <digital_constellation.h>
#include <gr_agc2_cc.h>
#include <digital_fll_band_edge_cc.h>
#include <digital_pfb_clock_sync_ccf.h>
#include <digital_constellation_receiver_cb.h>
#include <digital_diff_decoder_bb.h>
#include <gr_unpack_k_bits_bb.h>
#include <digital_map_bb.h>

class digital_generic_demod;
typedef boost::shared_ptr<digital_generic_demod> digital_generic_demod_sptr;
DIGITAL_API digital_generic_demod_sptr
digital_make_generic_demod(digital_constellation_bpsk_sptr constellation,
			   float samples_per_symbol, bool differential,
			   float excess_bw, bool gray_coded, float freq_bw,
			   float timing_bw, float phase_bw);

class DIGITAL_API digital_generic_demod : public gr_hier_block2
{
private:
  friend DIGITAL_API digital_generic_demod_sptr
    digital_make_generic_demod(digital_constellation_bpsk_sptr constellation,
			       float samples_per_symbol, bool differential,
			       float excess_bw, bool gray_coded, float freq_bw,
			       float timing_bw, float phase_bw);

  digital_constellation_bpsk_sptr d_constellation;
  float d_samples_per_symbol;
  bool d_differential;
  float d_excess_bw;
  bool d_gray_coded;
  float d_freq_bw;
  float d_timing_bw;
  float d_phase_bw;
  gr_agc2_cc_sptr d_agc;
  digital_fll_band_edge_cc_sptr d_freq_recov;
  digital_pfb_clock_sync_ccf_sptr d_time_recov;
  digital_constellation_receiver_cb_sptr d_receiver;
  digital_diff_decoder_bb_sptr d_diffdec;
  digital_map_bb_sptr d_symbol_mapper;
  gr_unpack_k_bits_bb_sptr d_unpack;

  digital_generic_demod(digital_constellation_bpsk_sptr constellation,
			float samples_per_symbol, bool differential,
			float excess_bw, bool gray_coded, float freq_bw,
			float timing_bw, float phase_bw);
};
