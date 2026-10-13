---
license: cc0-1.0
pretty_name: SFHQ part 3 - 750-image sample
size_categories:
- n<1K
tags:
- synthetic
- faces
- sfhq
- stylegan2
---

# SFHQ part 3: 750-image sample

A mirror of the 750 synthetic face images in the `a small samples (750 images)` folder of **Synthetic Faces High Quality (SFHQ) part 3**, by David Beniaguev. The images are unchanged: 1024×1024 JPEG files, stored in `images/` under their original file names.

The mirror exists so that the [Face Similarity Search Demo](https://huggingface.co/spaces/santiagodie/face-similarity-search-demo) can be rebuilt without downloading from Kaggle, which rate-limits anonymous downloads. It is built by the notebook `space/build_demo_index.ipynb` in [santiagodie/face-similarity-search-pipeline](https://github.com/santiagodie/face-similarity-search-pipeline), and `sfhq_pt3_sample_manifest.txt` lists the files.

## Source and license

- Original dataset: [SFHQ part 3 on Kaggle](https://www.kaggle.com/datasets/selfishgene/synthetic-faces-high-quality-sfhq-part-3), released under **CC0: Public Domain**.
- Project page: [github.com/SelfishGene/SFHQ-dataset](https://github.com/SelfishGene/SFHQ-dataset).
- Author: David Beniaguev. Credit goes to the original author; this repository only redistributes a subset.

## Caveats

- **Who declares the license:** the CC0 dedication is declared by the dataset author on Kaggle.
- **How the images were generated:** SFHQ part 3 was generated with StyleGAN2, a model trained on FFHQ (photos of real people from Flickr) and distributed by NVIDIA under a non-commercial license. That license covers the model and code; it does not explicitly cover generated images.
- **Resemblance to real people:** the faces are synthetic, but any of them can, by chance, resemble a real person.

This mirror is maintained for a non-commercial portfolio demo. If you believe an image should be removed, please open an issue in the GitHub repository.
