# Frozen real Russian OCR sample

`references.json` is the original visually transcribed reference, unchanged
from the first diagnostic on 1 November 2026. Its SHA-256 is
`db1c672c5ba070155209738d6169f0967eadc5ec287b6649831fc4e6dee3f07e`.
`sources.json` records the six original file hashes, byte sizes and download URLs.
The qualification refuses changed reference or source bytes.

The sample covers six selected pages of two financial reports and four images.
It checks 101 financial values, 16 subject/grade pairs, 12 receipt fields and
five text regions. It is a development regression sample, not a held-out
accuracy estimate. Neither the reference nor the limits may be changed to
accommodate recognition errors.

A subsequent visual source review found a transcription error: Rosseti page 5,
row `1180`, third amount is visibly `75 393`, while the original reference says
`76 393`. The original reference remains byte-for-byte unchanged.
`reference_adjudications.json` records the original and reviewed values, exact
source hash, page and visual evidence. This file is sealed in each reviewed run.
`--use-source-review` applies only these explicit corrections to a copy and saves
the original reference scores too. It does not change regions, limits, exclusions
or the number of controls. The recognizer never loads either reference file.

Source attribution:

- [Rosseti Yantar: appendix to the 2021 annual report](https://www.rosseti-yantar.ru/upload/iblock/c2f/Prilozhenie-k-Godovomu-otchetu-za-2021-g.-_Bukhgalterskaya-otchetnost_.pdf), pages 1, 5 and 7.
- [Bank Khlynov: 2014 annual report](https://www.bank-hlynov.ru/files/aktsionery-i-investory/%D0%93%D0%BE%D0%B4%D0%BE%D0%B2%D0%BE%D0%B9_%D0%BE%D1%82%D1%87%D0%B5%D1%82_2014_1.pdf), pages 3, 5 and 6.
- [Omar-2002/ru_benchmark_data](https://huggingface.co/datasets/Omar-2002/ru_benchmark_data/tree/6cc8ed1c3345db0874e27944c8d3d034fd342bd0), images `000003.jpg` and `000017.jpg` in `non_handwritten_real_data/images`.
- [cdek-ocr/receipt-ocr-ru](https://huggingface.co/datasets/cdek-ocr/receipt-ocr-ru/tree/50a7383534b3fe498c49eff9e7bb35b20f300eba), `images/test/IMG_0745_bin.jpg` and `IMG_0750_bin.jpg`. Dataset OCR annotations were not treated as ground truth.

PDFs and images are not redistributed here. Consult the source cards for their
terms. Redacted personal fields, handwriting, signatures, stamps, QR codes,
illegible tiny print and two damaged item amounts are outside the controls.
The first receipt's VAT is the visually printed `15.64`, not a calculated tax.

Download the exact files to a separate directory, retaining the filenames from
`sources.json`, or reuse the original local source directory. Run from the repo:

```powershell
uv run --no-sync python -m scripts.qualify_real_russian_ocr --sources output/real-russian-ocr-2026-11-01/sources --artifacts-path model_artifacts/ru --output output/real-russian-check/results --use-source-review
```

Use a new output directory. A nonzero exit code means that quality controls did
not all pass, even if every conversion returned `success`. Inference blocks
Python socket connections; native-library network traffic is not separately captured.
