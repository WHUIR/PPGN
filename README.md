
# PPGN

Codes for CIKM 2019 paper [Cross-Domain Recommendation via Preference Propagation GraphNet](https://doi.org/10.1145/3357384.3358166).

## Citation

Please cite our paper if you find this code useful for your research:

```
@inproceedings{cikm19:ppgn,
  author    = {Cheng Zhao and
               Chenliang Li and
               Cong Fu},
  title     = {Cross-Domain Recommendation via Preference Propagation GraphNet},
  booktitle = {The 28th ACM International Conference on Information and Knowledge Management, {CIKM} 2019, Beijing, China,
               November 3-7, 2019},
  pages     = {2165--2168},
  year      = {2019}
}
```

## Requirement
* Python 3.6
* Tensorflow 1.10.0
* Numpy
* Pandas
* Scipy


## Files in the folder
- `data/`
  - `data_prepare.py`: constructing cross-domain scenario from overlapping users;
  - `dataset.py`: defining the class of cross-domain dataset;
- `runner/`
  - `main.py`: the main function (including the configurations);
  - `model.py`: the detail implementation of PPGN;
  - `train.py`: training and evaluation;
- `utils/`
  - `metrics.py`: evaluation metrics.


## Running the code
1. Download the original data from [Amazon-5core](http://jmcauley.ucsd.edu/data/amazon/index.html), 
choose two relevant categories (*e.g.*, Books, Movies and TV) and put them under the same directory in data/.

2. run python data_prepare.py.

3. run python main.py.
