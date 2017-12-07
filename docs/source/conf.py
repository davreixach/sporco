# -*- coding: utf-8 -*-
#
# SPORCO documentation build configuration file, created by
# sphinx-quickstart on Tue Apr  7 06:02:44 2015.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import sys
import os
from builtins import next
from builtins import filter
from ast import parse
import re, shutil, tempfile

sys.path.append(os.path.dirname(__file__))
import callgraph

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath('../..'))


# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.mathjax',
    'numpydoc',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinxcontrib.bibtex',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.graphviz',
    'sphinx_tabs.tabs',
    'sphinx_fontawesome'
]

# generate autosummary pages
autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The encoding of source files.
source_encoding = 'utf-8'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'SPORCO'
copyright = u'2015-2017, Brendt Wohlberg'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
with open(os.path.join('../../sporco', '__init__.py')) as f:
    version = parse(next(filter(
        lambda line: line.startswith('__version__'),
        f))).body[0].value.s
# The full version, including alpha/beta/rc tags.
release = version

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
#today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['tmp', '*.tmp.*', '*.tmp']

# The reST default role (used for this markup: `text`) to use for all
# documents.
#default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
#add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
#modindex_common_prefix = []

# If true, keep warnings as "system message" paragraphs in the built documents.
#keep_warnings = False


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#html_theme = 'default'
#import sphinx_rtd_theme
#import sphinx_readable_theme
#html_theme = "sphinx_rtd_theme"
#html_theme = "bizstyle"
html_theme = "haiku"
#html_theme = "agogo"
#html_theme = 'readable'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#html_theme_options = {}

# Add any paths that contain custom themes here, relative to this directory.
#html_theme_path = []
#html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
#html_theme_path = [sphinx_readable_theme.get_html_theme_path()]

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
#html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
#html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
#html_logo = None

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
#html_favicon = None

# The style sheet to use for HTML and HTML Help pages. A file of that name
# must exist either in Sphinx' static/ path, or in one of the custom paths
# given in html_static_path.
html_style = 'sporco.css'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Add any extra paths that contain custom files (such as robots.txt or
# .htaccess) here, relative to this directory. These files are copied
# directly to the root of the documentation.
#html_extra_path = []

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
#html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
#html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
#html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
#html_additional_pages = {}

# If false, no module index is generated.
#html_domain_indices = True

# If false, no index is generated.
#html_use_index = True

# If true, the index is split into individual pages for each letter.
#html_split_index = False

# If true, links to the reST sources are added to the pages.
#html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
#html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
#html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
#html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
#html_file_suffix = None

# Output file base name for HTML help builder.
htmlhelp_basename = 'SPORCOdoc'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
# The paper size ('letterpaper' or 'a4paper').
#'papersize': 'letterpaper',

# The font size ('10pt', '11pt' or '12pt').
#'pointsize': '10pt',

# Additional stuff for the LaTeX preamble.
#'preamble': '',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
  ('index', 'SPORCO.tex', u'SPORCO Documentation',
   u'Brendt Wohlberg', 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#latex_use_parts = False

# If true, show page references after internal links.
#latex_show_pagerefs = False

# If true, show URL addresses after external links.
#latex_show_urls = False

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
#latex_domain_indices = True

#mathjax_path = 'MathJax/MathJax.js?config=default'
mathjax_path = 'https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML'

# Intersphinx mapping
intersphinx_mapping = {'http://docs.python.org/': None,
                       'http://docs.scipy.org/doc/numpy/': None,
                       'http://docs.scipy.org/doc/scipy/reference/': None,
                       'http://matplotlib.sourceforge.net/': None,
                       'http://hgomersall.github.io/pyFFTW/': None
                      }
# Added timeout due to periodic scipy.org down time
#intersphinx_timeout = 30

numpydoc_show_class_members = False

graphviz_output_format = 'svg'
inheritance_graph_attrs = dict(rankdir="LR", fontsize=9, ratio='compress',
                               bgcolor='transparent')
inheritance_node_attrs = dict(shape='box', fontsize=9, height=0.4,
                              margin='"0.08, 0.03"', style='"rounded,filled"',
                              fillcolor='"#f4f4ffff"')


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ('index', 'sporco', u'SPORCO Documentation',
     [u'Brendt Wohlberg'], 1)
]

# If true, show URL addresses after external links.
#man_show_urls = False


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
  ('index', 'SPORCO', u'SPORCO Documentation',
   u'Brendt Wohlberg', 'SPORCO', 'SParse Optimization Research COde (SPORCO)',
   'Miscellaneous'),
]

# Documents to append as an appendix to all manuals.
#texinfo_appendices = []

# If false, no module index is generated.
#texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
#texinfo_show_urls = 'footnote'

# If true, do not generate a @detailmenu in the "Top" node's menu.
#texinfo_no_detailmenu = False


on_rtd = os.environ.get('READTHEDOCS') == 'True'

if on_rtd:
    print('Building on ReadTheDocs')
    print
    print("Current working directory: {}" . format(os.path.abspath(os.curdir)))

if on_rtd:

    import matplotlib
    matplotlib.use('agg')

    if sys.version[0] == '3':
        from unittest.mock import MagicMock
    elif sys.version[0] == '2':
        from mock import Mock as MagicMock
    else:
        raise ImportError("Can't determine how to import MagicMock.")

    class Mock(MagicMock):
        @classmethod
        def __getattr__(cls, name):
            return MagicMock()

    MOCK_MODULES = ['pyfftw']
    sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

# See https://developer.ridgerun.com/wiki/index.php/How_to_generate_sphinx_documentation_for_python_code_running_in_an_embedded_system

# Sort members by type
#autodoc_member_order = 'groupwise'
autodoc_member_order = 'bysource'
#autodoc_default_flags = ['members', 'inherited-members', 'show-inheritance']


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
#exclude_patterns = ['_build', '**tests**', '**spi**']


# Ensure that the __init__ method gets documented.
def skip_member(app, what, name, obj, skip, options):
    if name == "__init__":
        return False
    if name == "IterationStats":
        return True
    if name == "timer":
        return True
    return skip


def process_docstring(app, what, name, obj, options, lines):
    if "IterationStats." in name:
        print("------> %s" % name)
        for n in xrange(len(lines)):
            print(lines[n])
        #for n in xrange(len(lines)):
        #    lines[n] = ''


def process_signature(app, what, name, obj, options, signature,
                      return_annotation):
    if "IterationStats." in name:
        print("%s : %s, %s" % (name, signature, return_annotation))


# See http://stackoverflow.com/questions/4427542
def rmsection(filename, pattern):

    pattern_compiled = re.compile(pattern)
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
        with open(filename) as src_file:
            for line in src_file:
                (sline, nsub) = pattern_compiled.subn('', line)
                tmp_file.write(sline)
                if nsub > 0:
                    next(src_file)
    shutil.copystat(filename, tmp_file.name)
    shutil.move(tmp_file.name, filename)



# See https://github.com/rtfd/readthedocs.org/issues/1139
def run_apidoc(_):

    # Import the sporco.admm and sporco.fista modules and undo the
    # effect of sporco.util._module_name_nested so that docs for
    # Options classes appear in the correct locations
    import inspect
    import sporco.admm
    for mnm, mod in inspect.getmembers(sporco.admm, inspect.ismodule):
        for cnm, cls in inspect.getmembers(mod, inspect.isclass):
            if hasattr(cls, 'Options') and inspect.isclass(getattr(cls,
                                                'Options')):
                optcls = getattr(cls, 'Options')
                if optcls.__name__ != 'Options':
                    delattr(mod, optcls.__name__)
                    optcls.__name__ = 'Options'

    import sporco.fista
    for mnm, mod in inspect.getmembers(sporco.fista, inspect.ismodule):
        for cnm, cls in inspect.getmembers(mod, inspect.isclass):
            if hasattr(cls, 'Options') and inspect.isclass(getattr(cls,
                                                'Options')):
                optcls = getattr(cls, 'Options')
                if optcls.__name__ != 'Options':
                    delattr(mod, optcls.__name__)
                    optcls.__name__ = 'Options'

    import sphinx.apidoc
    module = '../../sporco' if on_rtd else 'sporco'
    cpath = os.path.abspath(os.path.dirname(__file__))
    opath = cpath

    # Insert documentation for inherited solve methods
    callgraph.insert_solve_docs()

    # Remove auto-generated sporco.rst, sporco.admm.rst, and sporco.fista.rst
    rst = os.path.join(cpath, 'sporco.rst')
    if os.path.exists(rst):
        os.remove(rst)
    rst = os.path.join(cpath, 'sporco.admm.rst')
    if os.path.exists(rst):
        os.remove(rst)
    rst = os.path.join(cpath, 'sporco.fista.rst')
    if os.path.exists(rst):
        os.remove(rst)

    # Run sphinx-apidoc
    print("Running sphinx-apidoc with output path " + opath)
    sys.stdout.flush()
    sphinx.apidoc.main(['sphinx-apidoc', '-e', '-d', '2', '-o', opath,
                        module, os.path.join(module, 'admm/tests'),
                        os.path.join(module, 'fista/tests')])

    # Remove "Module contents" sections from specified autodoc generated files
    rmmodlst = ['sporco.rst', 'sporco.admm.rst', 'sporco.fista.rst']
    for fnm in rmmodlst:
        rst = os.path.join(cpath, fnm)
        if os.path.exists(rst):
            rmsection(rst, r'^Module contents')



def gencallgraph(_):

    print('Constructing call graph images')
    cgpth = '_static/jonga' if on_rtd else 'docs/source/_static/jonga'
    callgraph.gengraphs(cgpth, on_rtd)



def setup(app):
    app.connect("autodoc-skip-member", skip_member)
    app.connect('builder-inited', run_apidoc)
    #app.connect('autodoc-process-docstring', process_docstring)
    #app.connect('autodoc-process-signature', process_signature)

    gencallgraph(None)
