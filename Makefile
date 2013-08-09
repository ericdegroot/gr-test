CFLAGS=-pg -I/usr/local/include/gnuradio -I/usr/local/include

all: test

test: src/test.o src/packet_utils.o src/digital_packet_encoder.o src/digital_packet_decoder.o src/gr_complex_dump_cb.o src/digital_payload_source.o src/digital_generic_mod.o src/digital_generic_demod.o
	g++ -o test $^ $(CFLAGS) -L/usr/local/lib -lboost_system -lgnuradio-core -lgnuradio-digital -lgnuradio-uhd -pthread

src/test.o: src/test.cc
	g++ -c $< -o $@ $(CFLAGS)

src/gr_complex_dump_cb.o: src/gr_complex_dump_cb.cc src/gr_complex_dump_cb.h
	g++ -c $< -o $@ $(CFLAGS)

src/digital_payload_source.o: src/digital_payload_source.cc src/digital_payload_source.h
	g++ -c $< -o $@ $(CFLAGS)

src/digital_packet_encoder.o: src/digital_packet_encoder.cc src/digital_packet_encoder.h
	g++ -c $< -o $@ $(CFLAGS)

src/digital_packet_decoder.o: src/digital_packet_decoder.cc src/digital_packet_decoder.h
	g++ -c $< -o $@ $(CFLAGS)

src/digital_generic_mod.o: src/digital_generic_mod.cc src/digital_generic_mod.h
	g++ -c $< -o $@ $(CFLAGS)

src/digital_generic_demod.o: src/digital_generic_demod.cc src/digital_generic_demod.h
	g++ -c $< -o $@ $(CFLAGS)

src/packet_utils.o: src/packet_utils.cc src/packet_utils.h
	g++ -c $< -o $@ $(CFLAGS)

src/packet_utils_test.o: src/packet_utils_test.cc src/packet_utils.o
	g++ -c $< -o $@ $(CFLAGS)

packet_utils_test: src/packet_utils_test.o src/packet_utils.o
	g++ -o $@ $^ -L/usr/local/lib -lboost_system -lgnuradio-core -lgnuradio-digital -pthread

#src/gr_framer_sink_1.o: src/gr_framer_sink_1.cc src/gr_framer_sink_1.h
#	g++ -c $< -o $@ $(CFLAGS)
#
#src/digital_correlate_access_code_bb.o: src/digital_correlate_access_code_bb.cc src/digital_correlate_access_code_bb.h
#	g++ -c $< -o $@ $(CFLAGS)
#
#src/gr_message_source.o: src/gr_message_source.cc src/gr_message_source.h
#	g++ -c $< -o $@ $(CFLAGS)

clean:
	rm src/*.o test packet_utils_test
