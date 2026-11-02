# Implementation and experiment scope

The public layout reorganizes existing experiment components. It does not change
the original working folders, histories, checkpoints or private runners.
It has not undergone post-edit execution, import checks or numerical validation.

## Model interface

Every model returns `(logits, features)`, with `features['feats']`,
`features['preact_feats']`, and `features['pooled_feat']`. Original architecture
modules and state-dict keys are retained. CIFAR factories return the model through
a registry; ImageNet uses the original project ResNet and MobileNet implementations.
The two registries are loaded under separate package names to avoid import collisions.

## ALTER

The response, affine/blur transformation and competence curriculum are copied
from the experiment snapshot `ALTER2` into public `Distiller/ALTER.py`. The legacy
standalone wrapper is omitted; the public `DistillationObjective` composes ALTER
with a named existing distiller. No new standalone ALTER-only training objective
is exposed. Baseline uses original images only. The combined condition averages
CE over original and augmented images, keeps native distillation on the original
images, and adds the response objective with coefficient 200.

Response descriptors are final spatial backbone features averaged globally and
L2-normalized. The loss averages instance self-response squared discrepancy and
teacher-proximity-weighted log-strain squared discrepancy, with epsilon 1e-5.
Teacher and student process identical image contents; no trainable response
projection is added. The teacher remains frozen and in evaluation mode.

The transformation uses rotation ±7s degrees, scale [1-.05s,1+.05s], translation
±.07s in normalized grid coordinates, and a fixed 3x3 Gaussian blur selected
independently with probability .5 and blended with weight s. Strength starts at
.25, is bounded by [.25,1], and uses EMA beta=.99 and task calibration factor 1.10.

## Native objectives and weights

| Method | CE | Distillation | Preserved project implementation |
|---|---:|---:|---|
| KD | 1 | 16 | Logits KL, T=3 |
| DKD | 2 | 1 | alpha=3, beta=6, T=4, 20-epoch warmup, logits clamped to [-10,10] |
| RKD | 1 | .25 | distance=25, angle=50, on pooled features |
| FitNet | 2 | 100 | hint index 2, ConvReg with resize fallback |
| MLKD | .5 | 8 | MLKD_align_3, T=4/3/2, reduced-scalar confidence masking |
| ReviewKD | 1 | 1 | dynamic ABF/HCL, 20-epoch warmup |
| SKD | .1 | .9 | project SKD, T=4, tik=.1, sqrt floor 1e-12 |
| InDistill | 1 | 16 | stagewise intermediate MSE warmup followed by KD, T=3 |

For CIFAR MobileNetV1/ShuffleNetV2 students, MLKD defaults to distillation=.5,
following the recorded stability preset. An explicit `--distill-weight` overrides
this. CLI overrides apply equally to baseline and ALTER when `--compare` is used.
Original experiment configuration snapshots are included in `CIFAR-100/configs`;
the public runner documents its own defaults rather than claiming that every
historical tuned run used one common coefficient.

ReviewKD retains convolutional ABF branches and HCL; differing stage counts use
normalized-depth matching. Adaptive-average HCL bins use explicit mean reductions
to avoid unsupported deterministic CUDA backward. SKD keeps the project's scalar
reduction/masking convention. These are environment adaptations.

InDistill is a **classification stage-port**: ordered stages use 6/7/8 epochs
(plus 9 for VGG's fourth stage), followed by CE and logits KD. Fixed L1 filter
scores select target channels without physically pruning the teacher graph.
Five heterogeneous pairs first train one 2x-width student-topology auxiliary
teacher for 240 epochs with CE=1/KD=16/T=3. Both conditions share that teacher and
their intermediate warmup. ALTER is disabled during the intermediate-only stage.
No claim of full convolution-by-convolution upstream pruning reproduction is made.

## Protocol and output

The CIFAR dataset is split by `numpy.default_rng(2021)` into 45k/5k stratified
train/validation indices (50 validation examples per class). Base augmentation
is the project's weak/strong concatenation with its original RandAugment
implementation, including its sampling conventions. Validation does not select
checkpoints. Test accuracy is evaluated at the fixed final epoch.

The two conditions start from identical freshly initialized student weights.
They use deterministic per-index base augmentation and epoch order. FP32 is used
throughout; finite losses and gradients are enforced. CUDA determinism uses
warn-only mode; bitwise cross-platform reproducibility is not guaranteed.
Source/configuration hashes prevent silent resume under a changed implementation.

The release provides new public training entry points and a simpler checkpoint
format; it does not silently resume historical private-runner checkpoints.
Model state dicts remain loadable when architectures match.

ImageNet entry points share the public objectives but have a separate simple
ImageFolder protocol. They do not package specialized historical distributed,
Gaudi, overlap, or stability-repair launchers. InDistill is not enabled there.
