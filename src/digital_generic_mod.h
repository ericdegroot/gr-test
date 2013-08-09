#include <digital_api.h>
#include <gr_core_api.h>
#include <gr_hier_block2.h>

#include <digital_constellation.h>
#include <gr_packed_to_unpacked_bb.h>
#include <digital_map_bb.h>
#include <digital_diff_encoder_bb.h>
#include <digital_chunks_to_symbols_bc.h>
#include <gr_pfb_arb_resampler_ccf.h>

class digital_generic_mod;
typedef boost::shared_ptr<digital_generic_mod> digital_generic_mod_sptr;
DIGITAL_API digital_generic_mod_sptr
digital_make_generic_mod(digital_constellation_bpsk_sptr constellation,
			 float samples_per_symbol, bool differential,
			 float excess_bw, bool gray_coded);

class DIGITAL_API digital_generic_mod : public gr_hier_block2
{
private:
  friend DIGITAL_API digital_generic_mod_sptr
    digital_make_generic_mod(digital_constellation_bpsk_sptr constellation,
			       float samples_per_symbol, bool differential,
			       float excess_bw, bool gray_coded);

  digital_constellation_bpsk_sptr d_constellation;
  float d_samples_per_symbol;
  bool d_differential;
  float d_excess_bw;
  bool d_gray_coded;
  gr_packed_to_unpacked_bb_sptr d_bytes2chunks;  
  digital_map_bb_sptr d_symbol_mapper;
  digital_diff_encoder_bb_sptr d_diffenc;
  digital_chunks_to_symbols_bc_sptr d_chunks2symbols;
  gr_pfb_arb_resampler_ccf_sptr d_rrc_filter;

  digital_generic_mod(digital_constellation_bpsk_sptr constellation,
		      float samples_per_symbol, bool differential,
		      float excess_bw, bool gray_coded);
};
