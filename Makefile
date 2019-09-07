
FILENAME:=$(patsubst img/%.jpg, digits/%.jpg, $(wildcard img/*.jpg))

all: clean getimages $(FILENAME)

digits/%.jpg: img/%.jpg
	python recognize.py extract-digits $< $@

getimages:
	rsync -a pidom1:~/gazmeter/img/ img/

clean:
	rm -rf digits
	mkdir digits

downloadfroms3:
	echo "Run something like aws s3 cp s3://metering-vanves/img img --recursive --exclude '*' --include 'image-2019*-0900*jpg'"