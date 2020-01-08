# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = dowhy
SOURCEDIR     = source
BUILDDIR      = ../../dowhy-docs
PDFBUILDDIR   = /tmp
PDF           = ../manual.pdf

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile


latexpdf:     
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(SOURCEDIR) $(PDFBUILDDIR)/latex     
	#                                          ^^^
	@echo "Running LaTeX files through pdflatex..."
	make -C $(PDFBUILDDIR)/latex all-pdf     
	#         ^^^
	cp $(PDFBUILDDIR)/latex/*.pdf $(PDF)     
	#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^     
	@echo "pdflatex finished; see $(PDF)"

commithtmldocs: html
	cp -r $(BUILDDIR)/html/* $(BUILDDIR)
	cd $(BUILDDIR); git add . ; git commit -m "rebuilt docs"; git push origin gh-pages

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
