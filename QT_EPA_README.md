# QT-EPA (experimental)
This branch extends the user-owned DSER-QQ implementation without changing original `master`.

## Model
Use `--qt-epa` to enable **all three** levels of residual deformable-offset refinement. Each branch uses an explicit query fraction, ordered early/late voxel features, 5x5 local correlation, and a bounded gated 18-channel offset residual. The output head is zero-initialized to reproduce original pretrained predictions before fine-tuning.

## Train / fine-tune
```bash
python train_bsergb.py --data-root /data/BSERGB/3_TRAINING --pretrained checkpoints/bsergb_eventaid.pth --allow-partial-pretrained --qt-epa --amp --output-dir train/qt_epa
```
For baseline, omit `--qt-epa` and `--allow-partial-pretrained`.
Resume after fine-tuning:
```bash
python train_bsergb.py --data-root /data/BSERGB/3_TRAINING --qt-epa --resume train/qt_epa/last.pth --amp --output-dir train/qt_epa
```
`--resume` restores optimizer/scheduler; `--pretrained` only loads network weights. Do not mix.

## Eval
```bash
python interpolation.py --dataset bsergb --data_root_path /data/BSERGB/1_TEST --save_root_path eval/qt_epa --checkpoints_path train/qt_epa/last.pth --qt-epa
```
When QT-EPA is enabled, all evaluation scripts must provide `tau`. At present BS-ERGB is wired; other datasets must be verified separately.

## Caveats
- `tau` uses frame-index ratios, valid for evenly spaced frame times. Replace with timestamp ratios if needed.
- GitHub static editing cannot demonstrate CUDA DeformConv runtime or PSNR improvement; test on the actual GPU.
- Matching uses signed voxel values and positive/negative polarity cancellation remains possible.
- The branch is an experiment, not an established paper innovation; run baseline, query-only, correlation-only, and gate ablations.
