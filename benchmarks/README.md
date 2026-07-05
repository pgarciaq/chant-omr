# Benchmarks

This directory contains manually transcribed (image, GABC) pairs for evaluating
the trained model on real historical manuscripts.

## Creating benchmark data

1. Process book photographs through ghh (Stages 0-7 minimum) to get
   clean, dewarped page images.

2. For each page, manually transcribe the Gregorian chant into GABC notation.
   Use [GABC documentation](https://gregorio-project.github.io/gabc/index.html)
   as reference.

3. Save pairs as:
   ```
   benchmarks/
     lpa1/
       page_001.png     # dewarped page image from ghh
       page_001.gabc    # manual GABC transcription
       page_002.png
       page_002.gabc
       ...
     lpa2/
       ...
   ```

4. Aim for 20-30 pages per book covering diverse content (simple chants,
   complex melismas, different clefs, page layouts with varying staff count).

## Metrics

- **GABC Edit Distance (GED)**: Character-level Levenshtein distance between
  predicted and ground-truth GABC, normalized by reference length. Lower is
  better. Analogous to Transcoda's OMR-NED metric.

- **Neume Accuracy**: Accuracy on neume groups only (content inside parentheses),
  ignoring text syllables. Measures musical content recognition specifically.

- **Structural Validity**: Percentage of predictions that parse as valid GABC
  (balanced parentheses, valid clef declarations, etc.).
