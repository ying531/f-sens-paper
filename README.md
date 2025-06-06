<h1 align="center">
<p> f-sens-paper
</h1>


Reproduction code for paper [Sensitivity analysis under the f-sensitivity models: a distributional robustness perspective](https://arxiv.org/abs/2203.04373), to appear at Operations Research, 2025+.


### Files

The code for reproducing the simulation results in Section 5 is in the folder `/code`.

- `fix.py` runs the experiments in Section 5.2 with a fixed $\delta$ and varying $\rho$ (one seed). The results will be stored in a newly created folder `./fix_results/`
- `simu.py` runs the experiments in Section 5.3 with varying $(\delta,\rho)$ (one seed). The results will be stored in a newly created folder `./results/`
- After running the above experiments for all seeds and parameter configurations, `plot.R` produces the figures in Section 5.



#### Reference

```
@article{jin2022sensitivity,
  title={Sensitivity analysis under the $ f $-sensitivity models: a distributional robustness perspective},
  author={Jin, Ying and Ren, Zhimei and Zhou, Zhengyuan},
  journal={arXiv preprint arXiv:2203.04373},
  year={2022}
}
```