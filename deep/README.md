# Deep project directory

## TODO
- [ ] Implement different models (Dynamic)
- [X] Get good image shape and, if necessary, interpolation function
- [ ] Missing neutral class
- [ ] Set grid parameters to test
- [ ] Train on valeria

## Description
It is in this folder that we will implement the training loop and make the deep learning experiments

### Parameters
- MODEL: EffiecentNetV2 (S, M), ViT(32 and 16), ConvNeXT (Tiny, Small)
- Dataset size: 5%, 10%, 30%, 60%, 100%

## How to use
```bash
python main.py --experiment=experiment1 --config=configs/E_s.yml --debug --cpu --fract=0.1
```

## Examples
