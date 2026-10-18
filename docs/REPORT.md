# Experimental report

This report explains what was measured, how, and what the results mean. Every number here comes
from `results/*.json`; the complete generated tables (with confidence intervals, provenance and
figures) are in [`REPORT_TABLES.md`](REPORT_TABLES.md). To regenerate everything, run
`scripts/reproduce.sh`.

## 1. Setup

**Hardware.** Laptop, 12-thread x86-64 CPU, NVIDIA RTX 4050 Laptop GPU (6 GB), 16 GB RAM.
Embedding uses the GPU in fp32 with TF32 disabled (CPU- and GPU-computed vectors agree to
4.4e-6 max abs difference; checked on 64 images). FAISS runs on the CPU.

**Models** (pinned Hub revisions, `configs/models.yaml`):

| Key | Model | Space | Dim | Text encoder |
|---|---|---|---|---|
| `clip-vit-b32` | OpenAI CLIP ViT-B/32 | CLIP-B/32 | 512 | yes |
| `clip-vit-b16` | OpenAI CLIP ViT-B/16 | CLIP-B/16 | 512 | yes |
| `clip-vit-l14` | OpenAI CLIP ViT-L/14 (224 px) | CLIP-L/14 | 768 | yes |
| `dinov2-small` | DINOv2 ViT-S/14, CLS token | DINOv2-S | 384 | no |
| `dinov2-base` | DINOv2 ViT-B/14, CLS token | DINOv2-B | 768 | no |
| BLIP ITM | `Salesforce/blip-itm-base-coco` (reranker only) | — | — | cross-encoder |

**Data.** Flickr30k with the Karpathy split (test 1,000 images / 5,000 captions; val 1,014 / 5,070;
all 28,994 train images as distractors after deduplication), and Imagewoof (val 3,928 for
evaluation; a seeded 1,050-image train subset for tuning). See the README for acquisition,
licences and deduplication.

**Protocol.**

1. All tuning (fusion weights, reranker hyper-parameters, Platt scaling and background
   statistics) uses Flickr30k **val** or Imagewoof **train** only. Test is evaluated once per
   configuration; nothing was chosen by looking at it.
2. Exact (flat) search unless the experiment is about indexing.
3. Uncertainty: 95% percentile-bootstrap confidence intervals (1,000 resamples). For text→image,
   resampling is **clustered by target image**, because an image's five captions are not
   independent queries. Differences between configurations use a paired bootstrap on the same
   queries (2,000 resamples) and report the mean difference, its CI and a two-sided p-value.
4. Ranking depth is capped at 1,000 for R@K/MRR tasks (this changes MRR by < 0.001 per query);
   category-level image retrieval ranks the full gallery because mAP needs it.

## 2. Tasks and relevance labels

| Task | Queries | Gallery | Relevant | Metrics |
|---|---|---|---|---|
| T→I (1K) | 5,000 test captions | 1,000 test images | the caption's image (exactly 1) | R@1/5/10, MRR, median rank |
| T→I (full) | 5,000 test captions | 30,000 images (test + train distractors) | same | same |
| I→T | 1,000 test images | 5,000 test captions | the image's 5 captions | R@K, MRR, **mAP** |
| I→I category | 3,928 Imagewoof val images (leave-one-out) | the other 3,927 | same breed (~390 each) | **mAP**, P@1, P@10 |
| I→I instance | 1,000 transformed Flickr test images | 30,000 images | the untransformed original | R@1/10, MRR, by transformation |
| Hybrid (Imagewoof) | breed-name prompt + one image of that breed | val images | same breed | mAP, P@1, P@10 |
| Hybrid (Flickr30k) | first caption + 15–25%-area crop of the target | 31,014 images | the target | R@1/5/10 |

With exactly one relevant item, AP equals reciprocal rank, so mAP is only reported where queries
have several relevant items. Text→image and image→image results are never pooled: they use
different labels and answer different questions.

<!-- RESULTS -->
