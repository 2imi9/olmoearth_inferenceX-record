---
license: other
license_name: olmoearth-artifact-license
license_link: https://huggingface.co/2imi9/OlmoEarth-v1_2-FT-AWF-Base/blob/main/LICENSE.txt
base_model: allenai/OlmoEarth-v1_2-Base
datasets:
- allenai/olmoearth_projects_awf
---
This model is fine-tuned from [OlmoEarth-v1_2-Base](https://huggingface.co/allenai/OlmoEarth-v1_2-Base/) to map land use and land cover in southern Kenya.
It follows the recipe and labels of Ai2's [OlmoEarth-v1-FT-AWF-Base](https://huggingface.co/allenai/OlmoEarth-v1-FT-AWF-Base/), with the v1.2 encoder in place of v1.
It is a community fine-tune, not an Ai2 release.

Validation accuracy: 90.1% on the 344 validation points of the AWF dataset at the checkpoint kept (epoch 60 of 100), which was chosen on the same points, as Ai2's 89.5% for v1 was; 89.0% at the last epoch.

Training configuration: [model.yaml](https://github.com/2imi9/olmoearth_inferenceX/blob/3847ae7d55bd6e3f48e908603dfa313518db55fd/exp/jobs/e102_config/model.yaml), Ai2's AWF model.yaml with the encoder set to `OLMOEARTH_V1_2_BASE` and a CSV logger in place of Weights & Biases (rslearn 0.1.17, seed 0).
`model.ckpt` is the checkpoint `rslearn model fit` wrote; use it with `rslearn model predict --config model.yaml --ckpt_path model.ckpt`.

This model is licensed under the [OlmoEarth Artifact License](LICENSE.txt).
It is intended for use in accordance with Ai2's
[Responsible Use Guidelines](https://allenai.org/responsible-use).

For more details, please see the documentation on GitHub at https://github.com/allenai/olmoearth_projects/blob/main/docs/awf.md
