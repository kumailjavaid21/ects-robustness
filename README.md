# Robust Early Classification of Time Series under Corruption

**Working title:** Representational Robustness for Early Classification of Time Series

This is a results-and-code snapshot for a study of early classification under sensor corruption. The results retain the original analysis scope and limitations; they are not a claim that the stopping rule provides a finite-sample error guarantee.

## Results

| Directory | Contents |
| --- | --- |
| [frontier_isoearliness](results/frontier_isoearliness/) | Eight-dataset fixed-prefix comparisons and corruption-agnostic stopping frontiers. |
| [scale_26datasets](results/scale_26datasets/) | The 26-dataset benchmark, metrics, matched-earliness analyses, statistical tests and critical-difference figures. |
| [selective_and_ece](results/selective_and_ece/) | Fixed operating-point analysis, selective/abstaining classifiers and expected calibration error. |
| [deep_backbone/pilot](results/deep_backbone/pilot/) | Eight-dataset causal TCN and weight-inheritance pilot. |
| [deep_backbone/full_scale](results/deep_backbone/full_scale/) | 26-dataset causal TCN and missingness-mask comparison. |
| [crosstype_transfer](results/crosstype_transfer/) | Evaluation on unseen corruption types and mixed-versus-matched calibration transfer. |
| [baseline_v2](results/baseline_v2/) | Earlier MiniRocket baseline summary. |
| [robustness_v2](results/robustness_v2/) | Earlier eight-dataset, five-draw realistic-versus-oracle calibration study. |

Each directory retains its run's summary, all CSV files and PNG figures with their original relative paths. Read the summaries for dataset counts, definitions, caveats and statistical families. In particular, some matched-earliness comparisons are descriptive or have unequal realized decision times; five-draw standard deviations are not dataset-level confidence intervals. The crosstype results do not establish universal transfer.

## Data

The experiments use the UCR-2018 train/test splits from the [ChengsenWang/UCR-2018 dataset](https://huggingface.co/datasets/ChengsenWang/UCR-2018), accessed through the HF mirror. The scale study records dataset revision `289508ec44ac7086ca466689b2838fa8ba80ca57`. Raw data, cached features/predictions and fitted model weights are excluded. [MANIFEST.txt](MANIFEST.txt) lists excluded files and sizes.

## Pipeline and reproduction

[src/ects](src/ects/) contains the shared pipeline: data retrieval, causal running normalization, MiniRocket features with logistic classifiers at prefix checkpoints, corruption generators, stopping rules, and evaluation/report utilities. [src/experiments](src/experiments/) contains the relevant later run-specific workers and analyses. The benchmark uses clean, Gaussian-noise and random-missing conditions, paired draws, and a mixed-corruption calibration set; the cross-type study evaluates separately specified unseen corruptions. Details and exceptions are in each run's summary.

The recorded scale environment used Python 3.10.22, aeon 1.3.0, scikit-learn 1.7.2, NumPy 1.26.4, SciPy 1.15.3, pandas 2.3.3 and PyArrow 25.0.1; deep experiments used PyTorch 2.4.1+cu121. No environment lockfile is provided. Reproduction requires obtaining the UCR data, rebuilding the omitted caches/models, and adapting the run-specific scripts' original project paths and output layout. This snapshot provides the evidence and source, not a turnkey complete rerun.

No manuscript files are included.
