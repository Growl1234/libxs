# LIBXS Documentation

# Overview

* **ReadtheDocs**: [main](https://libxs.readthedocs.io/) and sample documentation with full text search.
* **PDF**: [main](https://github.com/hfp/libxs/raw/main/documentation/libxs.pdf) documentation file, and separate [sample](https://github.com/hfp/libxs/raw/main/documentation/libxs_samples.pdf) documentation.

## Building Documentation

```bash
cd libxs

# Build static documentation and PDFs
make documentation

# Serve with live reload
make mkdocs
```

## Building Presentations

```bash
cd libxs

# Serve with live reload
mkslides serve

# Build static HTML
mkslides build -d site/slides
```
