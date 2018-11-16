
FILENAME:=$(patsubst img/%.jpg, digits/%.jpg, $(wildcard img/*.jpg))

all: clean getimages $(FILENAME)

digits/%.jpg: img/%.jpg
	python recognize.py extract-digits $< $@

getimages:
	rsync -a pidom1:~/gazmeter/img/ img/

clean:
	rm -rf digits
	mkdir digits
