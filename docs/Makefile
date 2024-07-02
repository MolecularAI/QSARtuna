# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build

# Manually-created sources for Sphinx.
SOURCESOURCEDIR = sphinx-source
#
# # Auto-generated and manually-created sources for Sphinx.
SOURCEDIR     = sphinx-staging
#
# # Output from Sphinx.
BUILDDIR      = sphinx-builddir

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

html: Makefile
	mkdir -p $(SOURCEDIR)/ && cp -rf $(SOURCESOURCEDIR)/* $(SOURCEDIR)/
	mkdir -p $(SOURCEDIR)/docs/images && cp -rf images/* $(SOURCEDIR)/docs/images/
	cp -f ../README.md $(SOURCEDIR)/README.md
	mkdir -p $(SOURCEDIR)/notebooks && cp -rf ../notebooks/* $(SOURCEDIR)/notebooks/
	mkdir -p $(SOURCEDIR)/tests/data/DRD2/subset-50 && cp -rf ../tests/data/DRD2/subset-50/* $(SOURCEDIR)/tests/data/DRD2/subset-50
	PYTHONPATH=../ sphinx-apidoc -o $(SOURCEDIR) ../optunaz
	$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

$(SOURCEDIR)/_static/pytest:
	cd .. && pytest --cov-report html:docs/$(SOURCEDIR)/_static/pytest/coverage/ --cov-report xml:coverage.xml --html=docs/$(SOURCEDIR)/_static/pytest/pytest/index.html --cov optunaz

scpdocdeploy:
	rsync -avz $(BUILDDIR)/html/ login.scp.astrazeneca.net:/projects/mai/public_html/qsartuna/ --no-o --no-g --omit-dir-times --no-perms
	# adding --delete to rsync removes qsartuna/releases!
	# Alternative:
	# scp -r $(BUILDDIR)/html login.scp.astrazeneca.net:/projects/mai/public_html/qsartuna/

clean:
	rm -rf $(BUILDDIR) $(SOURCEDIR)
