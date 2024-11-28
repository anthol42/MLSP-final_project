# Deep project directory

## TODO
- [ ] Implement different models (Dynamic)
- [X] Get good image shape and, if necessary, interpolation function
- [ ] Missing neutral class
- [ ] Set grid parameters to test
- [ ] Train on valeria
- Try 1D CNN
- Try Finetuning head of transformer
- Try with promptable LLM (Fine tune LLAMA, zero shot prompting)
	- With Images
	- With numbers

## Feedback
- [ ] Expliquer davantage ce qu’est un alpha
- [ ] Work with synthetic data (generate time series 99 and 1 are targets) linear function with noise
    - [ ] Plot accuracy in function of noise
- [ ] Do grad Cam saliency maps
- [ ] Test for different domain generalization. Train on one market, then test on another market (Ex: train on US, then test on TW)
- [ ] Inject vector representation to LSTM or Transformer + time series(or use attention)

## Description
It is in this folder that we will implement the training loop and make the deep learning experiments

### Parameters
- MODEL: EffiecentNetV2 (S, M), ViT(32 and 16), ConvNeXT (Tiny, Small)
- Effect of substract or add (Image generation) on performances. (For best model)
- Dataset size: 5%, 10%, 30%, 60%, 100% (For best model and best image generation technique)

## How to use
```bash
python main.py --experiment=experiment1 --config=configs/E_s.yml --debug --cpu --fract=0.1
```

## Examples
