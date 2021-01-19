## Wasserstein Natural Policy Gradients (WNPG) + Wasserstein Natural Evolution Strategies (WNES) 

Implementations of WNPG and WNES from our paper [Efficient Wasserstein Natural Gradients for Reinforcement Learning](https://arxiv.org/abs/2010.05380)



To run WNPG: 

```python
python run_ppo_kwng.py
```

To run WNES, see `WNES.ipynb`.



Requirements: 

- [TensorFlow](https://www.tensorflow.org/) >= 1.14
- [MuJoCo](https://www.roboti.us/license.html)
- [OpenAI Gym](https://gym.openai.com/)
- [OpenAI Baselines](https://github.com/openai/baselines)
- [PyBullet Gym](https://github.com/benelot/pybullet-gym)



If you find this code useful, it would be great if you could cite us using: 

```
@misc{moskovitz2020efficient,
      title={Efficient Wasserstein Natural Gradients for Reinforcement Learning}, 
      author={Ted Moskovitz and Michael Arbel and Ferenc Huszar and Arthur Gretton},
      year={2020},
      eprint={2010.05380},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

