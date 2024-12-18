# Deep project directory
This directory contains the code and the data to train, test and analyse the different experiments presented in the 
paper. To reproduce the results, follow the **Get Started** instructions, then the **Reproduce the results** section.
To simply explore the code and test few hypothesis, you can read the **Single Run** section, which explain how the 
project works and give an explanation of each cli parameters.

## Get Started
First, create a virtual environment and install the dependencies. Assuming you are on linux, or using WSL and your working
directory is the current directory, run this:
```shell
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Then, you need to generate the result table. The result table is a file that, when used from the API, is immutable. It
automatically saves the results of each experiment with the given parameters. You only need to create it once. Run:
```shell
python buildResultTable.py
```

Once this is done, the rest of the setup, such as the dataset download and the directory creations, is automatically 
handled by the training script:) You can pass to the **Single Run** section to have a glance on how to run the project.

## Single Run
Every experiment, except the RF, is run through the ```main.py``` file. An example is:
```shell
python main.py --experiment=experiment1 --config=configs/E_s.yml --debug --cpu --fract=0.1
```
The mandatory parameters are: ```experiment``` and ```config```. In this project, there are three experiments, each 
described in their respecting markdown file: [experiment1](./experiments/experiment1.md), [experiment2](./experiments/experiment2.md),
[experiment3](./experiments/experiment3.md). Some config are designed for a specific experiment, so make sure to use compatible configurations.
The project uses tensorboard, so the evolution of the different metrics are saved in the runs folder. The ResultTable 
automatically assigns an id to each run, so this id will be used as name in the tensorboard console.

### Parameters
- fract: When specified, the float number between 0 and 1 is used as a fraction of the dataset. For example, 10% of the dataset is used in the previous example. This can speed-up the training time.
- comment: Add a comment to the run. It is added to the result table.
- dataset: Which dataset to use
- watch: Which metric to watch to save the best model.
- task: Which task to perform (The mechanism used to annotate the samples)
- split: How to split the data (along time, stocks, or random)

### Flags
- debug: Run in debug mode. The runID associated will be DEBUG and the results won't be saved in the result table
- cpu: Run only on cpu
- sample_inputs: Sample a batch of data and saves it to the model directory specified in the config.
- noscaler: Deactivate the mixed precision training. If set, it will train only in 32 bit precision.


## Reproduce the results
The official result table obtained by the paper is in the file resultTable.json.bak and CompiledResults.txt.bak. The 
objective behind eahc run is noted in the table of the respecting markdown file of the experiment. Note that some
runs appears in the resultTable, but aren't specified. They might be runs that have been started by mistake or run with 
bugs that were found afterwards. Note that the resultTable is immutable to ensure the results authenticity and 
reproducibility, so we did not removed them. So, to reproduce the results, you can run the following script, that will 
run all the experiments runs. However, the runID won't match the original run ids. So, to reproduce the figures, you 
will need to map the old runIDs to your new ones.
```shell
./scripts/reproduce.sh
```

Next, you can run the training of the random forest. Run:
```shell
python test_RF.py
```

Next, to get the naive performances of the different datasets, run:
```shell
python get_naive_perf --config=configs/paper_2.yml --dataset=<the dataset> --fract=0.1 or 1.
```
The dataset can be: _huge_, _small_ or _smallUS_, equivalent to: all us stocks, TW50 and US50 respectively. To get the 
same results as in the paper, set the fract to 0.1 when testing the huge dataset, and 1 otherwise. The pre-computed naive
performances are already noted in the [notebooks/ablation.ipynb](notebooks/ablation.ipynb) notebook.

The notebook [notebooks/ablation.ipynb](notebooks/ablation.ipynb) contains all the code used to generate the figures.
If you want to reproduce the figures from the paper, you will need to change the runIDs to match you runIDs. Next,
you can simply run each cells.

## Next steps
These are interesting future direction that we would have took if we had more time. We did not mention them in the paper,
and did not implemented them. However, we plan to test them.
- [ ] Work with synthetic data (generate time series 99 samples and 1 are targets) linear/sigmoid functions with noise
    - [ ] Plot accuracy in function of noise
    - [ ] We will see the capacity of our model to handle noise
- [ ] Do grad Cam saliency maps to understand what makes the model make X decision.
- [ ] Test for different domain generalization. Train on one market, then test on another market (Ex: train on US, then test on TW)
- [ ] Make the model multimodal: Inject a vector representation encoded by our vision network to a LSTM or Transformer with the stock price time series

