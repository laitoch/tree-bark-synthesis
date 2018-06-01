all: clean build

build:
	mkdir -p output_p log
	./run.py ai.* >/dev/null 2>/dev/null &

clean:
	rm -rf output_p/ log/
	rm -f a.png bi.png b.png fdm.png foobar.png quant_no_lonely.png

.PHONY: all build clean
