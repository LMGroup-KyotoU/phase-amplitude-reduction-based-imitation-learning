# Latent Dynamics Learning via Phase-Amplitude Reduction for Human-to-Robot    
This repository is the code implementation for the [Phase-amplitude reduction-based imitation learning](https://doi.org/10.1080/01691864.2024.2441242).
The trajectory-based imitation learning is powerfull method to transfer the human dynamic movement to the robot.
The phase amplitude reduction based dynamical model provide the novel latent dynamical system learning method and decompose the nonlinear dynamic behavior to the phase and amplitude component.

# Get started
## installation 

```
pip install -r requirements.txt
```
We tested codes in python version `3.10`.
The freezed package list is in file `requirements.lock` in our test environment.

# Notebook exmpale 
### simple limit cycel task
The pre-trained model file is saved in `data`.
If you want to re-try the training, please remove the comment out before load model file.  

1. [limitcycle_phase_eq.ipynb](notebooks/limitcycle_phase_eq.ipynb)
1. [Rossler-Equation-Identify.ipynb](notebooks/Rossler-Equation-Identify.ipynb)
1. [Stuart-Landau-Identify.ipynb](notebooks/Stuart-Landau-Identify.ipynb)
1. [Loss ablation study](notebooks/loss_ablation_result.ipynb)

# Citation
```
@article{Yamamori2024Phase,
    author = {Satoshi Yamamori and Jun Morimoto},
    title = {Phase-amplitude Reduction-based Imitation Learning},
    journal = {Advanced Robotics},
    volume = {0},
    number = {0},
    pages = {1--15},
    year = {2024},
    publisher = {Taylor \& Francis},
    doi = {10.1080/01691864.2024.2441242},
    URL = {https://doi.org/10.1080/01691864.2024.2441242},
    eprint = {https://doi.org/10.1080/01691864.2024.2441242}
}
```