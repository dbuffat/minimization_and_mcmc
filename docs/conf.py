# -- Project information -----------------------------------------------------
project = 'TP2 - Inférences et MCMC'
copyright = '2024, MT & DB'
author = 'MT & DB'
release = 'fr'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
]
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
language = 'fr'

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'  # ou 'sphinx_rtd_theme', 'pydata_sphinx_theme' #'furo'
html_static_path = ['_static']
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.ico"
html_css_files = ['custom.css']

# Thème options
html_theme_options = {
    "navigation_depth": 4,
    "show_prev_next": True,
}

# Autodoc configuration
autodoc_default_options = {
    'members': True,            # Document all members
    'private-members': True,    # Include private members
    'special-members': True,    # Include special members (e.g. __init__)
    'exclude-members': '',      # Exclude specific members, leave empty for none
    'undoc-members': False      # Do not include undocumented members
}
autoclass_content = "both"  # Include both class docstring and __init__ docstring
autodoc_member_order = "bysource"  # Order members by their appearance in the source code

