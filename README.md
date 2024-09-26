# MLSP-final_project
The sound of money

## Sonification
1. For each day in the window:
   - Use the relative height of the historical price of another window as the frequency and convert it to 'air pressure' unsing a reverse fourier transform
   - This should give a sound of maybe 1 to 2 second that will be considered as a 'note'
3. Concatenate the notes of every days of the window to make a 'melody'
4. Feed the sound to a machine learning algorithm to do the classification.  (Usually requires to compute the spectrogram)
