# MLSP-final_project
The sound of money

## Sonification
1. For each day in the window:
   - Use the relative height of the historical price of another window as the frequency and convert it to 'air pressure' unsing a reverse fourier transform
   - This should give a sound of maybe 1 to 2 second that will be considered as a 'note'
3. Concatenate the notes of every days of the window to make a 'melody'
4. Feed the sound to a machine learning algorithm to do the classification.  (Usually requires to compute the spectrogram)


## Roadmap
- [ ] Implemention of code [8h actif: 27 Novembre]
- [ ] Hyperparameter search and training [~1 week passif: 3 Novembre]
- [ ] Analysis and choosing the best model (Making figures for ablation study) [~6h actif: 17 Novembre]
- [ ] Backtest [2h actif + 1 week passif: 22 novembre]
- [ ] Write the paper [16h actif (fin de semaine): 1er décembre]