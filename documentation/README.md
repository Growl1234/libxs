# LIBXS Documentation

## Documentation

* **ReadtheDocs**: [main](https://libxs.readthedocs.io/) and sample documentation with full text search.
* **PDF**: [main](https://github.com/hfp/libxs/raw/main/documentation/libxs.pdf) documentation file, and separate [sample](https://github.com/hfp/libxs/raw/main/documentation/libxs_samples.pdf) documentation.

```bash
make documentation    # Build PDFs
make mkdocs           # Serve with live reload
```

Building the PDFs requires Pandoc and LaTeX. Serving the documentation requires the packages in `.theme/requirements.txt` (`pip install -r .theme/requirements.txt`).

## Presentations

Slide decks built with [mkslides](https://github.com/MartenBE/mkslides) (Reveal.js).

| Presentation | File | Purpose |
| --- | --- | --- |
| Practical Ozaki Schemes | `ozaki/index.md` | Introduction, deployment, and MPI reference |
| Self-Diagnosing Parameter Prediction | `predict/index.md` | Confidence-gated prediction and deployment safety |

```bash
make mkslides                 # Serve Ozaki with live reload
make mkslides SLIDES=predict  # Serve Predict with live reload
```

Controls: `Space` next, `Esc` overview, `S` speaker notes, `F` fullscreen.
