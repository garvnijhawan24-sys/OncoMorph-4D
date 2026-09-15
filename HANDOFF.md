# Handoff notes — training the glioma segmentation U-Net

## What this is
Hackathon repo (2021, "Ingenium_NeurAi") for brain tumor segmentation + downstream
volumetric/survival analysis. We're adapting it to train on a new dataset,
**MU-Glioma-Post** (204 patients, longitudinal post-treatment glioma MRI, `PKG - MU-Glioma-Post/`),
since the original pretrained weights (`model_2.hdf5` referenced in the README's
Google Drive link) no longer exist there — only a screen-recording of someone
training the model live for 6 epochs (early-stopped) remains, and that run's
per-class dice was weak (necrotic 0.07, edema 0.19, enhancing 0.08). Not worth
recovering; better to train fresh on MU-Glioma-Post, which has real ground-truth
`tumorMask` files for all 204 patients.

## Where things stand
- [brain_tumor_segmentation_u_net.ipynb](brain_tumor_segmentation_u_net.ipynb) has
  been edited so its data-loading cells point at `PKG - MU-Glioma-Post/` instead of
  BraTS. Architecture, loss functions (dice/IoU/precision/sensitivity/specificity),
  and the training loop are untouched.
- Split is **patient-level** (not timepoint-level) — a patient's multiple
  timepoints never span train/val/test, to avoid leakage. 203 patients → 596
  patient-timepoint samples, split ~65/15/20.
- Modality mapping: dataset's `t2f` = BraTS `flair`, `t1c` = BraTS `t1ce` (the two
  channels the model actually uses). `tumorMask` uses the same 0/1/2/3 label
  scheme as BraTS `seg` (background/necrotic/edema/enhancing) — verified directly
  against the data.
- **Nothing has been trained yet.** The notebook is ready to run top to bottom.

## What still needs doing
1. Run [brain_tumor_segmentation_u_net.ipynb](brain_tumor_segmentation_u_net.ipynb)
   end to end (needs GPU — CPU epochs took 150–850s each in the original video,
   and this dataset is bigger).
2. Let it run the full scheduled 25 epochs (or adjust) rather than stopping at 6
   like the original run — that's what produced the earlier model's weak dice
   scores.
3. Save the resulting weights (`model.save_weights("model.h5")`, already in the
   notebook) somewhere durable.
4. Optionally: point
   [Volummetric_Analysis_and_3D_Reconstruction.ipynb](Volummetric_Analysis_and_3D_Reconstruction.ipynb)
   at the new weights + a MU-Glioma-Post case for the volumetric/3D-reconstruction
   pipeline (not yet adapted — still has BraTS-shaped paths).

## Setup
```bash
pip install -r requirements.txt
```
(The original `requirements.txt` in this repo's history was a dumped Python list,
not valid for pip — it's been fixed to real package names.)

Dataset lives in `PKG - MU-Glioma-Post/MU-Glioma-Post/PatientID_XXXX/Timepoint_Y/`,
each containing `*_brain_t1c.nii.gz`, `*_brain_t1n.nii.gz`, `*_brain_t2f.nii.gz`,
`*_brain_t2w.nii.gz`, `*_tumorMask.nii.gz`. Because it's 12 GB, it's not in git —
it's being transferred via USB and should land at `PKG - MU-Glioma-Post/` in the
repo root, matching the path the notebook expects (`DATASET_PATH` in the notebook).

## Hardware & gotchas
- **A GPU matters a lot here.** In the original walkthrough, epochs took
  150–850s each *on 369 BraTS cases*; this dataset has 596 patient-timepoint
  samples (more data), so CPU-only training could easily run overnight or
  longer for 25 epochs. If the training laptop has an NVIDIA GPU, make sure
  `tensorflow` picks it up (matching CUDA/cuDNN installed) before doing a
  full run — a quick `tf.config.list_physical_devices('GPU')` check up front
  saves a lot of wasted time.
- **Sanity-check on a tiny slice first.** Before committing to a full 25-epoch
  run, it's worth temporarily setting `epochs=1` (or slicing `train_ids[:5]`)
  just to confirm the pipeline runs end-to-end on this machine without
  crashing, before letting it run for real.
- **If you hit out-of-memory (GPU or RAM):** the `DataGenerator` loads 100
  full-resolution slices per sample per batch. First things to try, in order:
  lower `batch_size` in `DataGenerator(...)` calls (already 1 by default),
  or reduce `IMG_SIZE` (currently 128) — that also means editing the
  `input_layer = Input((IMG_SIZE, IMG_SIZE, 2))` cell consistently.
- Needs Jupyter (Notebook/Lab) or an editor that runs `.ipynb` cells (e.g.
  VS Code with the Jupyter extension) to actually execute this.
