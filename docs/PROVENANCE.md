# Source and initialization provenance

LeRobot is vendored under its Apache-2.0 license from local source commit
`bf31dd794ffb4f87380aba3912f64421e8352d3c` (version 0.6.2).
Only `src/`, its linked documentation, packaging metadata, README, and LICENSE are included, not experiment
outputs, environments, datasets, Git history, or training weights.

Four source files include existing local training fixes: `scripts/lerobot_train.py`,
`common/train_utils.py`, `utils/random_utils.py` (rank-specific RNG saving/restoring),
and `datasets/factory.py` (episode task metadata fallback). Unrelated RoboTwin,
RoboCasa and LIBERO-PRO modifications were NOT copied.

The entry wrapper isolates LoRA initialization RNG, preserves checkpoint
normalization statistics during fresh continuation, and uses the local tokenizer.
Both entry points filter synthetic empty camera entries only during camera-key
validation; it does not change actions or success predicates.

The exact dense initialization hashes are in `configs/initializations.json`.
All five originate from the main comparison. Base means the actual common theta0
after 200 LIBERO adaptation steps, **not an untouched official base**. TCR is c_e,
selected by its historical main-repeat score (322/400), not a development-only
selection. All three continued-training repeats start from this same TCR file.
Its historical score is not copied into this study's S(0), which must be reevaluated.

The old Base joint-training run (15k steps, one seed) is reference-only. The user
requested a new Base run under the same configuration as all four merger arms.
No single-expert arm is included. No upstream expert training is launched here.

Initial model weights remain unchanged. Publication sanitizes only the stale
`pretrained_path` in copied config.json. Source train_config.json and private paths
are not published. Tokenizer and processor files accompany each dense model.
The code license does not override the original model/tokenizer/data licenses.
