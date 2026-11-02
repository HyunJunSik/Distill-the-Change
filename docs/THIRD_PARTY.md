# Third-party source attribution

- Repository/README layout reference: [HyunJunSik/StreamLined](https://github.com/HyunJunSik/StreamLined).
- CNN model and common distiller foundations: [MEGVII MDistiller](https://github.com/megvii-research/mdistiller).
  The included source modules retain their existing attribution; see `MDistiller_LICENSE`.
- ReviewKD: [dvlab-research/ReviewKD](https://github.com/dvlab-research/ReviewKD),
  *Distilling Knowledge via Knowledge Review*, CVPR 2021. The project uses an
  adapted dynamic ABF/HCL implementation; see IMPLEMENTATION.md.
- SKD: [HyunJunSik/StreamLined](https://github.com/HyunJunSik/StreamLined),
  *Streamlined Knowledge Distillation*. The included core preserves the project's
  provided formulation and masking convention, not every upstream training detail.
- InDistill: [gsarridis/InDistill](https://github.com/gsarridis/InDistill),
  *InDistill: Information Flow-Preserving Knowledge Distillation for Model Compression*,
  WACV 2025. Reference commit: `70f47f063756f52e0bb33c3803350c597d258188`.
  The upstream Apache-2.0 license is retained at `Distiller/InDistill_LICENSE`.
- ALTER response loss, transformations and controller originate from the current
  project's frozen baseline-tuning experiment snapshot. Public composition,
  dataset registries and command-line entry points were reorganized for this release.

Files copied from third-party/project snapshots remain subject to their original
notices and terms. No blanket relicensing of upstream work is asserted. The
top-level license covers original release contributions to the extent the authors
hold rights in them; upstream components retain their respective licenses.
