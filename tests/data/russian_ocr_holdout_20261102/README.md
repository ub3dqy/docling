# Fresh Russian document check, 2 November 2026

This cohort was selected and visually transcribed before its first OCR run.
The recognizer is fixed at `653dc0c04dae1c95bc923006f3f57b3ef4d625df`.
No runtime, model, dependency, threshold or reference tuning is part of this check.
Once evaluated, these documents become known controls and must not be presented
as unseen documents in subsequent development checks.

There are ten page cases from seven distinct originals:

| Original | Selected pages | Controls |
| --- | --- | --- |
| Union of Cadastral Engineers, 2025 financial statements | 1, 2, 5, 6 | 192 table value cells, two headings |
| Union of Cadastral Engineers, 2025 auditor report | 1 | Heading and two full paragraphs |
| Clinic receipt, `IMG_0724_bin.PNG` | 1 | Seven numeric fields, two text regions, amount without VAT |
| Appliance receipt, `IMG_0872_bin.PNG` | 1 | Eight numeric fields, total and VAT text |
| Pharmacy receipt, `IMG_0891_bin.jpg` | 1 | Twelve numeric fields, payment and mixed VAT text |
| Open passport camera photograph, `000244.jpg` | 1 | Bilingual heading, issue/expiry dates and issuer |
| Birth certificate scanned image, `000300.jpg` | 1 | Two text regions, four numeric fields |

The controls comprise 192 table values (55 printed numbers and 137 dashes),
31 numeric fields and 14 text regions. Three numeric fields are printed VAT
amounts. Receipt photos were already binarized by their publisher. The original
PDF scan pages have no native text and contain page-sized raster images.
These are original published PDFs, not PDFs generated from the test photographs.

`references.json` stores the literal visual transcription, regions and a 3%
character error limit per text region. All numeric fields and selected table
values must match. Text normalization only applies NFC, case folding and
collapsed whitespace. Numeric table comparisons accept decimal comma/point,
thousands whitespace and equivalent negative notation. Numeric field checks
look for a complete number in both its region and the exported text; splitting
a number with spaces can fail that check. Region checks do not establish a
complete structured association between every item label and amount.
A dash is not a zero.
The page-2 figures `11`, `1388` and `1398` are intentionally preserved as
printed, although they do not add up. No expected value is calculated from
another amount. The recognizer does not read the expected answers.

`sources.json` binds the reference hash to the seven original file hashes,
byte sizes and stable source URLs. `novelty.json` records the prior-source
comparison. The original local audit inspected 53 previous run manifests and
25 unique downloaded originals, including unused earlier candidates. None of
the new source URLs, filenames or file hashes occurs in that inventory.
Five new images were also compared with previous source images in four
rotations using a 256-bit difference hash, with visual source-identity checks.
This supports novelty against the available project history, not against every
document on the Internet or unknown model-training data.

Source attribution:

- [Union of Cadastral Engineers: published financial reports](https://www.srokadastr.ru/01.01.07.08/279), original files [2025 financial statements](https://www.srokadastr.ru/files/2699) and [2025 auditor report](https://www.srokadastr.ru/files/2700).
- [cdek-ocr/receipt-ocr-ru, fixed revision](https://huggingface.co/datasets/cdek-ocr/receipt-ocr-ru/tree/50a7383534b3fe498c49eff9e7bb35b20f300eba), the three listed files under `images/test/`. Its generated OCR annotations were not used as reference answers.
- [Omar-2002/ru_benchmark_data, fixed revision](https://huggingface.co/datasets/Omar-2002/ru_benchmark_data/tree/6cc8ed1c3345db0874e27944c8d3d034fd342bd0), the two listed files under `non_handwritten_real_data/images/`. Its supplied transcriptions were not used as reference answers.

Source PDFs/images are not redistributed in Git. See the source publishers for
their terms. Download the exact files into one directory using the names in
`sources.json`. Record a failed download rather than substituting another file.

Run the existing qualification unchanged, choosing this fixture directory:

```powershell
$env:PYTHONUTF8 = '1'
uv run --no-sync python -c "from pathlib import Path; import scripts.qualify_real_russian_ocr as q; q.FIXTURES = Path('tests/data/russian_ocr_holdout_20261102').resolve(); q.main()" --sources output/russian-ocr-holdout-2026-11-02/sources --artifacts-path model_artifacts/ru --output output/russian-ocr-holdout-replay/results
```

Use a new output directory. Run on CPU with four threads, the Russian profile,
a 180-second per-document timeout, local model assets, and Python network
connections blocked, as configured by the unchanged runner. A nonzero exit code
means at least one quality control failed; it does not by itself mean a crash.
Native-library network traffic is not separately captured.

The first run and before/after runtime/model hashes are retained locally under
`output/russian-ocr-holdout-2026-11-02/`. The small published `result.json` and
`REPORT.md` describe its outcome. `reference_review_notes.json` records three
post-inference control limitations, including two correct dates, a card-payment
region set too low and one ambiguous unit-price glyph. The original references,
scores and thresholds remain unchanged; no adjusted accuracy replaces them.
Handwriting, signatures, stamps, QR/barcodes,
authentication of the originals, covered details and text outside the stated
controls are not graded. This small, deliberately varied sample does not
estimate general production accuracy.

The subsequent development recheck is documented in [RECHECK.md](RECHECK.md)
and [recheck-result.json](recheck-result.json). The first-run files above remain
unchanged. These documents are now known development/regression cases.
