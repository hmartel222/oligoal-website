# OligoGym released dataset collection

Copied without modification from Roche/OligoGym commit
`97f5b9f58d9e445a8ecb0218037af7465c3708c0`, directory
`oligogym/resources/pkg_dataset/`, on 2026-11-14 at the repository owner's request.
The website includes four processed datasets containing 35,423 measurement rows.
The upstream registry and original metadata JSON for those four datasets are
retained. The manifest records source paths and SHA256 for the full upstream
collection.
This notice and manifest are local additions; the released files are unchanged.

Credit: Rachapun Rotrattanadumrong and Carlo De Donno, *OligoGym: Curated Datasets
and Benchmarks for Oligonucleotide Drug Discovery*, NeurIPS 2025 Datasets and
Benchmarks. Source: https://github.com/Roche/OligoGym . Each processed JSON retains
the original study/patent source attribution; cite these sources when using a dataset.

LICENSE.upstream.md is the unmodified Apache-2.0 license shipped in that source
revision. Its presence is not an independent determination of rights in underlying
third-party studies, patent data, or trained weights. No new project license is
assigned by this copy. Further public distribution should preserve source notices
and the original source-specific rights/provenance review.

Datasets are independent tasks, not a fully observed multi-endpoint molecular table.
Preserve x/HELM, processed y, y_raw, target IDs, SMILES, FASTA and optional assay
columns. Do not rescale y, fill missing target IDs, merge incompatible endpoints,
or correct conflicting TLR7/TLR8 descriptions silently. Original experimental split
memberships and wet-lab validation of generated molecules are not provided here.
