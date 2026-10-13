---
title: Face Similarity Search Demo
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.29.1
python_version: "3.12"
app_file: app.py
license: mit
pinned: false
short_description: Face search over synthetic faces with FaceNet and a VP-tree
---

# Face Similarity Search Demo

Live demo of the face search pipeline from [santiagodie/face-similarity-search-pipeline](https://github.com/santiagodie/face-similarity-search-pipeline). It finds the most similar faces with dlib detection, FaceNet embeddings and a VP-tree index.

**This demo uses synthetic faces only.** None of the indexed images shows a real person. The original project was built on a private press-photo dataset that is not published, for privacy reasons.

## What is indexed

- **Source faces:** the 750 faces in the `a small samples (750 images)` folder of SFHQ part 3 (listed in `sfhq_pt3_sample_manifest.txt` in the GitHub repo). They are mirrored, unchanged, in the [santiagodie/sfhq-pt3-sample-750](https://huggingface.co/datasets/santiagodie/sfhq-pt3-sample-750) dataset.
- **Variations:** 4 per face, so that every synthetic "identity" has several matches to find: a tighter crop, a slight rotation, a brightness change and a horizontal flip.
- **Processing:** every image goes through the same steps as the original pipeline:
  1. Downscale to at most 600 px.
  2. dlib HOG detection, with dlib CNN (MMOD) as a fallback.
  3. Crop with a 20% margin and resize to 160×160.
  4. 128-d FaceNet embedding (DeepFace).
  5. Index in a Euclidean VP-tree.
- **Threshold:** results are filtered at a distance of 9.655, the F1-optimal threshold from the original project's evaluation notebook. A face with no match below that distance returns no results.

Each result is labelled with its synthetic identity (the SFHQ file name) and its distance. The index is built by the Colab notebook `space/build_demo_index.ipynb` in the GitHub repo.

## How to use it

- **Cara recortada (cropped face):** upload a face crop, which is resized to 160×160 and searched directly. The examples are crops from the index.
- **Foto completa (full photo):** upload any photo. Every detected face, up to 10, is cropped and searched separately.

The interface is in Spanish, the original language of the project.

The Space runs on the free CPU tier, so the first query after a cold start takes longer while TensorFlow and the models load.

## Dataset and license

The faces come from **Synthetic Faces High Quality (SFHQ)** by David Beniaguev:

- Dataset: [SFHQ part 3 on Kaggle](https://www.kaggle.com/datasets/selfishgene/synthetic-faces-high-quality-sfhq-part-3), released under **CC0: Public Domain**.
- Project page: [github.com/SelfishGene/SFHQ-dataset](https://github.com/SelfishGene/SFHQ-dataset) (code under the MIT License).

**Caveats:**

- The CC0 dedication is declared by the dataset author on Kaggle.
- SFHQ part 3 was generated with StyleGAN2, a model trained on FFHQ (photos of real people from Flickr) and distributed by NVIDIA under a non-commercial license. That license covers the model and code; it does not explicitly cover generated images.
- Synthetic faces can, by chance, resemble real people.

This demo is a non-commercial portfolio project. If you believe an image should be removed, please open an issue in the GitHub repository.

The app code is released under the MIT License. The dlib MMOD face detector is downloaded at startup from [davisking/dlib-models](https://github.com/davisking/dlib-models), and the FaceNet weights are downloaded by [DeepFace](https://github.com/serengil/deepface).
