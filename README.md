# Self-Supervised-Pretraining-and-Fine-Tuning-Model-for-Medical-Image-Segmentation-
A two-stage self-supervised pipeline for brain tumour segmentation: a Masked Autoencoder pretrains a ViT encoder on unlabelled BraTS MRI volumes, then transfers to a ResNet34+U-Net (ResUNet) decoder via pseudo-labelling. Outperforms a from-scratch U-Net baseline, especially at 5% labelled data.
