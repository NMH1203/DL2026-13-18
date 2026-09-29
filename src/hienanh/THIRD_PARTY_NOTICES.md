# Source and model attribution

The project methodology comes from
[NMH1203/Low-Light-Image-Enhancement-and-Downstream-Recognition](https://github.com/NMH1203/Low-Light-Image-Enhancement-and-Downstream-Recognition).
A copy of the project README used during preparation is in
`references/project_README.md`.

The input is the user's local Roboflow ExDark v12 export:
[ExDark on Roboflow Universe](https://universe.roboflow.com/project-h68de/exdark-kd37x/dataset/12).
Its supplied `README.dataset.txt` states CC BY 4.0. Both source README files are
preserved in each processed dataset under the `SOURCE_` prefix. The underlying
dataset is [ExDark by Yuen Peng Loh and Chee Seng Chan](https://github.com/cs-chan/ExDark-Dataset).

The model implementation follows the official architectures and uses official
pretrained weights from Chongyi Li and coauthors:

- [Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE): Chunle Guo et al.,
  *Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement*, CVPR 2020.
- [Zero-DCE++](https://github.com/Li-Chongyi/Zero-DCE_extension): Chongyi Li,
  Chunle Guo, and Chen Change Loy, *Learning to Enhance Low-Light Image via
  Zero-Reference Deep Curve Estimation*, IEEE TPAMI, 2021,
  DOI: 10.1109/TPAMI.2021.3063604.

Both official repositories state that their code is for academic research under
CC BY-NC 4.0. Their original model sources are preserved under `references/` for
numerical compatibility tests. The local implementation adapts naming, removes
unused layers, supports CPU execution, and prevents border cropping. The
checkpoint-compatible mathematics remain unchanged.

The downloaded checkpoint SHA-256 values are:

```text
zerodce_Epoch99.pth
a4395acb874f320375d9704997cef874eaaaaa26a1777ceb29a92b70f74c3612

zerodcepp_Epoch99.pth
ca8855b90df9a80fa4195a831f33d3476b1964f787eb70602797c773067f3b84
```

Detection labels follow the
[Ultralytics YOLO detection dataset format](https://docs.ultralytics.com/datasets/detect/).
