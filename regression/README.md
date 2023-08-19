# Environment & Run

## Uninstall the following packages after environment creation
- pytorch
- pytorch-mutex
- torch-tb-profiler
- torchaudio
- torchvision

### Uninstall commands used. Different channels used for installation.

```shell
> conda remove pytorch pytorch-mutex torchaudio torchvision
> pip uninstall torch-tb-profiler
```

## Install the packages again with cuda compatibility

```shell
> conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c conda-forge 
> conda install -c pytorch pytorch-mutex 
> pip install torch-tb-profiler
```

---

## Activate Environment

```shell
> conda activate card
```

## Deactivate Environment

```shell
> conda deactivate
```

## Run Training command

```shell
> python main.py --device 1 --thread 4 --loss card_conditional --exp ./results/card_conditional_options_preds/100steps/nn/run_1/f_phi_prior_cat_f_phi --run_all --n_splits 20 --doc options_pred --config configs/calls_predStock.yml
```

#
