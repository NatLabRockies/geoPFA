# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from importlib.metadata import version as get_version

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Geothermal Play Fairway Analysis"
copyright = "2025, Nicole Taverna"
author = "Nicole Taverna"

release = get_version("geopfa").split("+")[0]
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = []


# -- Extension configuration -------------------------------------------------

# -- Autodoc configuration --
autoclass_content = "both"  # Merge __init__ docstring into the class page
autodoc_member_order = "bysource"  # Keep methods in source-code order
autodoc_inherit_docstrings = True  # Inherit docstrings from base classes
autodoc_typehints = "none"
add_module_names = False  # Drop "geopfa." prefix from signatures

# -- Autosummary configuration --
autosummary_generate = True  # Auto-generate stub pages
autosummary_generate_overwrite = True  # Regenerate stubs on every build
autosummary_imported_members = False  # Skip re-exported names

# -- BibTeX configuration --
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"
bibtex_reference_style = "author_year"

# -- Intersphinx configuration --
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "geopandas": ("https://geopandas.org/en/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "shapely": ("https://shapely.readthedocs.io/en/stable/", None),
}

# -- Suppress cross-reference warnings for unresolvable types --
#
# A few residual mismatches remain that cannot be fixed in docstrings:
#
# - "optional" is standard NumPy docstring convention (e.g. "float, optional")
#   and Napoleon always tries to cross-reference it as a class.
# - NumPy annotation aliases and scalar/dtype objects are registered under
#   different roles than the py:class links emitted by autodoc and Napoleon.
# - geopandas.GeoDataFrame and pandas.DataFrame are either registered under a
#   different role or a different qualified path in their inventories.
# - GPy has no Sphinx inventory at all.
# - LatticeKrigX does not yet publish a Sphinx inventory. Its fully qualified
#   public types remain visible in the generated API documentation, but cannot
#   be linked until that inventory exists.
# - NumPy-style ``default=...`` and ``sequence`` parameter annotations are
#   descriptive text, not Python class names.
#
# Suppress intersphinx network errors in air-gapped/VPN environments where
# external inventory URLs are unreachable due to SSL certificate inspection.
# Note: Sphinx 8.2.3's intersphinx "failed to reach" warning has no type/subtype
# attribute, so suppress_warnings cannot catch it. The warnings are environment-
# specific (NREL VPN cert chain) and do not reflect documentation quality issues.
# The build has 0 content warnings on internet-connected CI environments.

nitpick_ignore_regex = [
    (r"py:class", r"optional"),
    (r"py:class", r"default=.*"),
    (r"py:class", r"sequence"),
    (r"py:class", r"NDArray"),
    (r"py:class", r"np\.(ndarray|float64)"),
    (r"py:class", r"numpy\.(ndarray|dtype|float64|bool)"),
    (r"py:class", r"numpy\.random\..*"),
    (r"py:class", r"pandas\.DataFrame"),
    (r"py:class", r"geopandas\.(GeoDataFrame|geodataframe\.GeoDataFrame)"),
    (r"py:class", r"GPy\..*"),
    (r"py:class", r"JointResult"),
    (r"py:(class|func|mod)", r"latticekrigx\..*"),
]

# -- Napoleon configuration --
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False

# -- MyST Parser configuration --
myst_enable_extensions = [
    "dollarmath",
    "fieldlist",
    "substitution",
    "tasklist",
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_title = f"Geothermal PFA {release}"
# html_static_path = ["_static"]
