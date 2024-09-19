# From Words 🤬 to Wheels 🚗: Automated Style-Customized Policy Generation for Autonomous Driving

**Words2Wheels** is a framework that automatically customizes driving policies based on user commands. It employs a Style-Customized Reward Function to generate a Style-Customized Driving Policy.

*Note: The paper is currently under review. We have released the Reinforcement Learning and the Statistical Evaluation parts of the code. The full prompts will be released once the paper is accepted.*

## 🚀 Getting Started

### Prerequisites

- **Python 3.8** or higher.

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yokhon/Words2Wheels.git
   cd Words2Wheels
   ```

2. **Create a virtual environment and activate it**:

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install the required packages**:

   ```bash
   pip install -r requirements.txt
   ```

### Training

- **Train All Reward Functions**:

  ```bash
  python train.py --reward all
  ```

  This command trains each reward function in `db/reward/` with 5 different seeds, which may take **a few hours** in total.


- **Reduce Training Time**:

  To train with only 1 seed and reduce the training time, add the `--n_run 1` flag.


- **Training Output**:

  The resulting models will be saved in the directory `db/model/ppo/`.

### Simulation

- **Test the Trained Models**:

  ```bash
  python simulate.py --reward all
  ```

  The resulting statistical information will be stored in `db/stat/`.


[//]: # (## Repository Structure)

[//]: # ()
[//]: # (- `db/reward/`: Initial reward functions.)

[//]: # (- `db/model/ppo/`: Trained models.)

[//]: # (- `db/stat/`: Statistical evaluation results.)

[//]: # (- `train.py`: Script to train the models.)

[//]: # (- `simulate.py`: Script to test the models.)


## 🔖 Citation
If you find our paper and codes useful, please kindly cite us via:

```bibtex
@misc{han2024words,
  title={From Words to Wheels: Automated Style-Customized Policy Generation for Autonomous Driving},
  author={Xu Han and Xianda Chen and Zhenghan Cai and Pinlong Cai and Meixin Zhu and Xiaowen Chu},
  year={2024},
  eprint={2409.11694},
  archivePrefix={arXiv},
  primaryClass={cs.RO}
}
```
