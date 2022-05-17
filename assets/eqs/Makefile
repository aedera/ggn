SOURCE_DIR = $(abspath .)
FILES      = $(wildcard $(SOURCE_DIR)/*.tex)
TARGETS    = $(patsubst $(SOURCE_DIR)/%.tex, $(CURDIR)/%.pdf, $(FILES))
TARGETS   += $(patsubst $(SOURCE_DIR)/%.tex, $(CURDIR)/%.png, $(FILES))

all : $(TARGETS)

$(CURDIR)/%.png : $(CURDIR)/%.pdf
	convert -density 300 -quality 100 $< $@

$(CURDIR)/%.pdf : $(SOURCE_DIR)/%.tex
	pdflatex $<

clean :
	- rm -rf *.log
	- rm -rf *.aux
	- rm -rf *.pdf
	- rm -rf *.png
