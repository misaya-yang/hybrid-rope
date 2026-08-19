.PHONY: all quick clean fig
PYTHON?=python3
all:   ; @./compile.sh full
quick: ; @./compile.sh quick
clean: ; @./compile.sh clean
fig:   ; @$(PYTHON) figs/make_fig_identification.py && $(PYTHON) figs/make_fig_method_overview.py
