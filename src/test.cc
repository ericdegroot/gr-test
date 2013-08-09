#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>

#include <boost/crc.hpp>

#include <gr_top_block.h>
#include <gr_file_source.h>
#include <gr_file_sink.h>

#include <gr_core_api.h>
#include <gr_sync_block.h>
#include <gr_io_signature.h>

#include <gr_channel_model.h>

#include <gr_uhd_usrp_source.h>
#include <gr_uhd_usrp_sink.h>

#include "packet_utils.h"
#include "gr_complex_dump_cb.h"
#include "digital_packet_encoder.h"
#include "digital_packet_decoder.h"
#include "digital_generic_mod.h"
#include "digital_generic_demod.h"

#define DEBUG 0
#define VERBOSE 0

////////////////////////////////////////////////////////////////////////////////

class gr_test_block2;
typedef boost::shared_ptr<gr_test_block2> gr_test_block2_sptr;

class GR_CORE_API gr_test_block2 : public gr_block
{
private:
  gr_test_block2():
    gr_block("gr_test_block2",
	     gr_make_io_signature(1, 1, sizeof(char)),
	     gr_make_io_signature(1, 1, sizeof(char)))
  {
    /* NOP */
  }

  int general_work(int noutput_items, gr_vector_int &ninput_items,
		   gr_vector_const_void_star &input_items,
		   gr_vector_void_star &output_items)
  {
    int n_in = ninput_items[0];
    const char *in = (const char *) input_items[0];
    char *out = (char *) output_items[0];

    int consumed = n_in;

    consume_each(consumed);

    int produced = consumed;

    // fprintf(stderr, "packet_decoder.general_work: noutput_items=%d, ninput_items=%d, consumed=%d, produced=%d\n", noutput_items, n_in, consumed, produced);

    return produced;
  }

  void forecast(int noutput_items, gr_vector_int &ninput_items_required)
  {
    ninput_items_required[0] = noutput_items;
  }

  friend GR_CORE_API gr_test_block2_sptr gr_make_test_block2();
};

GR_CORE_API gr_test_block2_sptr gr_make_test_block2()
{
  return gnuradio::get_initial_sptr(new gr_test_block2());
}

////////////////////////////////////////////////////////////////////////////////

class gr_test_block;
typedef boost::shared_ptr<gr_test_block> gr_test_block_sptr;

class GR_CORE_API gr_test_block : public gr_sync_block
{
private:
  gr_test_block(size_t itemsize, std::string name):
    gr_sync_block("gr_test_block", gr_make_io_signature(1, 1, itemsize),
		  gr_make_io_signature(1, 1, itemsize)),
    d_itemsize(itemsize),
    d_name(name)
  {
    /* NOP */
  }

  int work(int noutput_items, gr_vector_const_void_star &input_items,
	   gr_vector_void_star &output_items)
  {
    const char *in = (const char *) input_items[0];
    char *out = (char *) output_items[0];

    // if (VERBOSE) {
      fprintf(stderr, "test_block[%s].work: noutput_items=%d\n", d_name.c_str(), noutput_items);

      // print stream in hex
      for (int i = 0; i < 80; i++) {
	fprintf(stderr, "%02X", (unsigned char) in[i]);
      }
      fprintf(stderr, "\n");
      // }

    std::memcpy(out, in, noutput_items * d_itemsize);

    return noutput_items;
  }

  friend GR_CORE_API gr_test_block_sptr
    gr_make_test_block(size_t itemsize, std::string name);

  size_t d_itemsize;
  std::string d_name;
};

GR_CORE_API gr_test_block_sptr
gr_make_test_block(size_t itemsize, std::string name)
{
  return gnuradio::get_initial_sptr(new gr_test_block(itemsize, name));
}

////////////////////////////////////////////////////////////////////////////////

class image_source;
typedef boost::shared_ptr<image_source> image_source_sptr;

class GR_CORE_API image_source : public gr_sync_block
{
private:
  image_source(const char* filename, bool repeat):
    gr_sync_block("image_source",
		  gr_make_io_signature(0, 0, 0),
		  gr_make_io_signature(1, 1, sizeof(char))),
    d_fileSize(0), d_fileContents(NULL), d_fileContentsSize(0),
    d_fileIndex(0), d_checksum(0), d_repeat(repeat)
  {
    std::ifstream file(filename, std::ios::in | std::ios::binary |
		       std::ios::ate);
    if (file.is_open()) {
      d_fileSize = file.tellg();
      file.seekg(0, std::ios::beg);

      d_fileContentsSize = d_fileSize + 0; // 13
      d_fileContents = new char[d_fileContentsSize];
      if (!file.read(d_fileContents + 0, d_fileSize)) {
	perror(filename);
      }

      file.close();
    }

    fprintf(stderr, "image_source: filename=%s, fileSize=%d\n", filename, d_fileSize);
    /*
    d_fileContents[0] = 0x54;
    d_fileContents[1] = 0x60;
    d_fileContents[2] = 0x00;
    d_fileContents[3] = 0x00;
    d_fileContents[4] = 0x00;
    */

    /*
    d_fileContents[0] = 0x54;
    d_fileContents[1] = 0x6F;
    d_fileContents[2] = 0x61;
    d_fileContents[3] = 0x53;
    d_fileContents[4] = 0x74;

    uint32_t n_fileSize = htonl(d_fileSize);

    d_fileContents[5] = ((n_fileSize >> 0) & 0xFF);
    d_fileContents[6] = ((n_fileSize >> 8) & 0xFF);
    d_fileContents[7] = ((n_fileSize >> 16) & 0xFF);
    d_fileContents[8] = ((n_fileSize >> 24) & 0xFF);

    boost::crc_32_type crc;
    crc.process_bytes(d_fileContents + 13, d_fileSize);
    d_checksum = crc.checksum();

    uint32_t n_checksum = htonl(d_checksum);

    d_fileContents[9] = ((n_checksum >> 0) & 0xFF);
    d_fileContents[10] = ((n_checksum >> 8) & 0xFF);
    d_fileContents[11] = ((n_checksum >> 16) & 0xFF);
    d_fileContents[12] = ((n_checksum >> 24) & 0xFF);
    */

    for(int i = 0; i < 22; i++) {
      fprintf(stderr, "%02X ", (unsigned char) d_fileContents[i]);
    }
    fprintf(stderr, "\n");
  }

  ~image_source()
  {
    /* NOP */
  }

  int work(int noutput_items, gr_vector_const_void_star &input_items,
	   gr_vector_void_star &output_items)
  {
    // const char *in = (const unsigned char *) input_items[0];
    char *out = (char *) output_items[0];
    int outputSize, outputRemaining, fileRemaining = 0, copyLength = 0;

    outputSize = std::min(noutput_items, (int) d_fileContentsSize);
    outputRemaining = outputSize;

    while (outputRemaining > 0) {
      fileRemaining = d_fileContentsSize - d_fileIndex;
      copyLength = std::min(outputRemaining, fileRemaining);
      std::memcpy(out, d_fileContents + d_fileIndex, copyLength * sizeof(char));
      out += copyLength;
      d_fileIndex += copyLength;
      /*
      if (d_fileIndex == d_fileContentsSize) {
	fprintf(stderr, "Leaving(-1)\n");
	return -1;
	//d_fileIndex = 0;
      }
      */
      outputRemaining -= copyLength;
    }

    // fprintf(stderr, "image_source.work: noutput_items=%d, d_fileIndex=%d, d_fileSize=%d, d_fileContentsSize=%d\n", noutput_items, d_fileIndex, d_fileSize, d_fileContentsSize);

    //fprintf(stderr, "Leaving\n");
    return outputSize;
  }

  friend image_source_sptr make_image_source(const char*, bool repeat);

  uint32_t d_fileSize;
  uint32_t d_fileContentsSize;
  char* d_fileContents;
  int d_fileIndex;
  uint32_t d_checksum;
  bool d_repeat;
};

image_source_sptr make_image_source(const char* filename, bool repeat)
{
  return gnuradio::get_initial_sptr(new image_source(filename, repeat));
}

////////////////////////////////////////////////////////////////////////////////

class image_sink;
typedef boost::shared_ptr<image_sink> image_sink_sptr;

class GR_CORE_API image_sink : public gr_sync_block
{
private:
  image_sink():
    gr_sync_block("image_sink",
		  gr_make_io_signature(1, 1, sizeof(char)),
		  gr_make_io_signature(0, 0, 0))
  {
    /* NOP */
  }

  int work(int noutput_items, gr_vector_const_void_star &input_items,
	   gr_vector_void_star &output_items)
  {
    const unsigned char *in = (const unsigned char *) input_items[0];
    // unsigned char *out = (unsigned char *) output_items[0];

    for (int i = 0; i < noutput_items; i++) {
      if (i < noutput_items - 8 && in[i] == 137 && in[i + 1] == 80 && in[i + 2] == 78 &&
          in[i + 3] == 71 && in[i + 4] == 13 && in[i + 5] == 10 && in[i + 6] == 26 &&
	  in[i + 7] == 10) {
	fprintf(stderr, "PNG Header Found\n");

	if (d_buffer.length() > 0) {
	  std::ofstream file("sink.png", std::ios::out | std::ios::binary);
	  if (file.is_open()) {
	    file.write(d_buffer.data(), d_buffer.length());
	    file.close();

	    fprintf(stderr, "\tsink.png written\n");
	  }

          d_buffer.erase();
	}
      }

      d_buffer.push_back(in[i]);
    }

    /*
    for (int i = 0; i < noutput_items - 5; i++) {
      if (in[i] == 'T' && in[i + 1] == 'o' && in[i + 2] == 'a' &&
	  in[i + 3] == 'S' && in[i + 4] == 't') {
	fprintf(stderr, "ToaSt Header Found\n");
      }
    }
    */

    // std::memcpy(out, in, noutput_items * sizeof(char));

    return noutput_items;
  }

  friend image_sink_sptr make_image_sink();

  std::string d_buffer;
};

image_sink_sptr make_image_sink()
{
  return gnuradio::get_initial_sptr(new image_sink());
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
  int samples_per_symbol = 2;
  int bits_per_symbol = 1;
  int samp_rate = 1e6; // 10000

  double center_freq = 910e6;
  double gain = 0;

  float excess_bw = 0.350f;
  float freq_bw = 2.0f*M_PI/100.0f; // 0.06280f; // 2*M_PI/100.0;
  float phase_bw = 2.0f*M_PI/100.0f; // 0.06280f; // 2*M_PI/100.0;
  float timing_bw = 2.0f*M_PI/100.0f; // 0.06280f; // 2*M_PI/100.0;

  int payload_len = 512;
  bool pad_for_usrp = false;
  int whitener_offset = 0;
  bool whitening = false;

  bool usrp_receiving = false;
  bool usrp_transmitting = false;

  int c;
  while ((c = getopt(argc, argv, "l:rtuw::")) != -1) {
    switch (c) {
    case 'l':
      payload_len = atoi(optarg);
      break;
    case 'r':
      usrp_receiving = true;
      break;
    case 't':
      usrp_transmitting = true;
      break;
    case 'u':
      pad_for_usrp = true;
      break;
    case 'w':
      whitening = true;

      if (optarg != NULL)
	whitener_offset = atoi(optarg);

      break;
    case '?':
      fprintf(stderr, "Option error\n");
      return 1;
    default:
      abort();
    }
  }

  char* filename;
  if (optind < argc) {
    filename = argv[optind];
  } else {
    fprintf(stderr, "No input file specified.\n");
    return 1;
  }

  // Construct a top block that will contain flowgraph blocks.  Alternatively,
  // one may create a derived class from gr_top_block and hold instantiated blocks
  // as member data for later manipulation.
  gr_top_block_sptr tb = gr_make_top_block("test");

  gr_file_source_sptr source = gr_make_file_source(sizeof(unsigned char),
  						   filename, true);
  digital_packet_encoder_sptr packet_encoder =
    digital_make_packet_encoder(payload_len, samples_per_symbol, bits_per_symbol,
				pad_for_usrp, whitener_offset, whitening);
  digital_packet_decoder_sptr packet_decoder =
    digital_make_packet_decoder(payload_len, whitener_offset, whitening);
  //gr_file_sink_sptr sink = gr_make_file_sink(sizeof(unsigned char), "sink1.png");
  //sink->set_unbuffered(true);

  digital_constellation_bpsk_sptr constellation = digital_make_constellation_bpsk();

  digital_generic_mod_sptr modulator =
    digital_make_generic_mod(constellation, samples_per_symbol, true,
			     excess_bw, true);

  gr_channel_model_sptr channel_model;
  if (!usrp_transmitting && !usrp_receiving) {
    std::vector<gr_complex> taps;
    std::complex<float> tap(1.0f, 1.0f);
    taps.push_back(tap);
    channel_model = gr_make_channel_model(0.01f, 0, 1.0f, taps, 42);
  }

  digital_generic_demod_sptr demodulator =
    digital_make_generic_demod(constellation, samples_per_symbol, true, excess_bw,
			       true, freq_bw, timing_bw, phase_bw);

  // image_source_sptr image_source = make_image_source(filename, false);
  image_sink_sptr image_sink = make_image_sink();

  boost::shared_ptr<uhd_usrp_sink> usrp_sink;
  if (usrp_transmitting) {
    std::string sink_device_addr("addr=192.168.10.3");
    usrp_sink = uhd_make_usrp_sink(sink_device_addr, uhd::stream_args_t("fc32"));
    usrp_sink->set_samp_rate(samp_rate);
    usrp_sink->set_center_freq(center_freq);
    usrp_sink->set_gain(gain, 0);
  }

  boost::shared_ptr<uhd_usrp_source> usrp_source;
  if (usrp_receiving) {
    std::string source_device_addr("addr=192.168.10.4");
    usrp_source = uhd_make_usrp_source(source_device_addr,
				       uhd::stream_args_t("fc32"));
    usrp_source->set_samp_rate(samp_rate);
    usrp_source->set_center_freq(center_freq);
    usrp_source->set_gain(gain, 0);
  }

  // Connect blocks
  if (!usrp_receiving) {
    tb->connect(source, 0, packet_encoder, 0);
    //tb->connect(image_source, 0, packet_encoder, 0);
    tb->connect(packet_encoder, 0, modulator, 0);

    if (usrp_transmitting) {
      printf("Connecting modulator to USRP sink\n");
      tb->connect(modulator, 0, usrp_sink, 0);
    } else {
      printf("Connecting modulator to channel model\n");
      tb->connect(modulator, 0, channel_model, 0);
    }
  }

  if (!usrp_transmitting) {
    if (usrp_receiving) {
      printf("Connecting USRP source to demodulator\n");
      tb->connect(usrp_source, 0, demodulator, 0);
    } else {
      printf("Connecting channel model to demodulator\n");
      tb->connect(channel_model, 0, demodulator, 0);
    }

    tb->connect(demodulator, 0, packet_decoder, 0);
    //tb->connect(packet_decoder, 0, sink, 0);
    tb->connect(packet_decoder, 0, image_sink, 0);
  }

  if (DEBUG) {
    if (!usrp_receiving) {
      gr_file_sink_sptr sink2 = gr_make_file_sink(sizeof(unsigned char),
						  "log/premod1");
      sink2->set_unbuffered(true);
      gr_file_sink_sptr sink4 = gr_make_file_sink(sizeof(gr_complex),
						  "log/midmod1");
      sink4->set_unbuffered(true);
      gr_file_sink_sptr sink5 = gr_make_file_sink(sizeof(unsigned char),
						  "log/preencoder1");
      sink5->set_unbuffered(true);

      // tb->connect(image_source, 0, sink5, 0);
      tb->connect(source, 0, sink5, 0);
      tb->connect(packet_encoder, 0, sink2, 0);
      tb->connect(modulator, 0, sink4, 0);
    }

    if (!usrp_transmitting) {
      gr_file_sink_sptr sink3 = gr_make_file_sink(sizeof(unsigned char),
						  "log/postmod1");
      sink3->set_unbuffered(true);
      tb->connect(demodulator, 0, sink3, 0);
    }
  }

  // Tell GNU Radio runtime to start flowgraph threads; the foreground thread
  // will block until either flowgraph exits (this example doesn't) or the
  // application receives SIGINT (e.g., user hits CTRL-C).
  //
  // Real applications may use tb->start() which returns, allowing the foreground
  // thread to proceed, then later use tb->stop(), followed by tb->wait(), to
  // cleanup GNU Radio before exiting.
  // tb->run();

  tb->start();

  char inputChar;
  while ((inputChar = getchar()) != 'q');

  tb->stop();
  tb->wait();

  // Exit normally.
  return 0;
}
