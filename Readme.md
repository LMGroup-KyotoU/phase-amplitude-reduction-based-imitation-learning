# Latent Dynamics Learning via Phase-Amplitude Reduction for Human-to-Robot    
This repository is the code implementation for the [Phase-amplitude reduction-based imitation learning](https://doi.org/10.1080/01691864.2024.2441242).
The trajectory-based imitation learning is a powerful method to transfer the human dynamic movement to the robot.
The phase amplitude reduction-based dynamical model provides the novel latent dynamical system learning method and decomposes the nonlinear dynamic behavior to the phase and amplitude component.

# Get started
## installation 

```
pip install -r requirements.txt
```
We tested codes in Python version `3.10`.
The frozen package list is in the file `requirements.lock` in our test environment.

We also prepared trained model files following [download link](https://drive.google.com/file/d/1ZEK94ZaQxnI86ZCczyJrsr_0kOy7e_wm/view?usp=drive_link).
If you get this model file, you can evaluate notebooks without full training.

# Notebook example 
### simple limit cycle task
The pre-trained model file is saved in `data`.
If you want to re-try the training, please remove the comment before loading the model file.

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
