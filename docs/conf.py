import sys
import os
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'xftsim'
copyright = '2023, Richard Border'
author = 'Richard Border'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
sys.path.insert(0, os.path.abspath(".."))


#html_logo = "_static/xftsimlogomedium.svg"
html_logo = "_static/xftsimlogomediumwhite.svg"
#html_logo = "_static/xftsimlogo.svg"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "myst_parser",
    #"sphinx_autosummary_accessors",
    #"sphinxawesome_theme",
    "sphinx.ext.extlinks",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "nbsphinx",
#    "sphinx_autosummary_accessors",
#    "sphinx.ext.linkcode",
#    "sphinx_copybutton",
    #"sphinxext.rediraffe",
    #"sphinx_design",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

myst_enable_extensions = ['attrs_inline', 'substitution']

autosectionlabel_prefix_document = True
autosummary_generate = True
autodoc_typehints = "none"

# Napoleon configurations

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = True

master_doc = "index"
exclude_patterns = ["_build", "**.ipynb_checkpoints"]
source_suffix = '.rst'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
#html_theme='press'
html_theme = 'sphinx_rtd_theme'
#html_theme = 'sphinx_book_theme'
#html_theme = 'sphinxawesome_theme'
html_static_path = ['_static']
html_theme_options = {
        'logo_only':True,
            "show_navbar_depth": 2,
    "repository_url": "https://github.com/rborder/xftsim/docs",
    "use_repository_button": True,
        }
html_css_files = ['custom.css']
