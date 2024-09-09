# Transformer Model Implementation

This repository is originally forked from [transformer repository by Hyunwoong in Pytorch](https://github.com/hyunwoongko/transformer) which contains an implementation of the Transformer model based on the paper **"Attention is All You Need"** by Vaswani et al. This model was the first to introduce the self-attention mechanism and has since become foundational in the field of NLP and deep learning.

In 

## Features

This repository extends the basic implementation of the Transformer with several additional features to improve usability, deployment, and monitoring:

- **Hydra for configuration**: Simplifies configuring hyperparameters and model settings.
- **Docker integration**: A `Dockerfile` is included to allow easy deployment and environment isolation.
- **TensorBoard support**: For logging and monitoring model training.
- **Requirements file**: A `requirements.txt` file is provided to specify the exact versions of the dependencies used.
- **Bug fixes**: Various bugs from the base implementation have been fixed.
- **Code restructuring**: The codebase has been reformatted for better readability and maintainability.
- **Progress tracking with tqdm**: tqdm has been integrated to display progress bars during training.

## Upcoming Features

We are thinking about continous improvement. Adding new features according to the recent tools and frameworks could be beneficial to have a better understanding about the concept we had previously.

- **Captum**: In order to have a better understanding of how the model (in our case transformer) is behaving behind the scene.

## Getting Started

### Prerequisites

To get started, ensure you have the following installed:

- Python 3.8 or higher
- Docker (optional, but recommended for deployment)
- TensorBoard (for monitoring training progress)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/transformer-implementation.git
   cd transformer-implementation
   ```
2. **Install the required dependencies:**

    Using `pip`:

    ```bash
    pip install -r requirements.txt
    ```
    Alternatively, you can use **Docker** to create a self-contained environment (instructions below).

### Configuration
The configuration system is managed via **Hydra**. To modify hyperparameters or settings, you can simply update the config.yaml file or pass them via command-line arguments.

```bash
python main.py --config-name config.yaml
```
For example, to override the batch size, you can run:

```bash
python main.py batch_size=32
```

### Docker
To build and run the model inside a Docker container:

1. **Build the Docker image:**

    ```bash
    docker build -t transformer-implementation --network=host .
    ```

2. **Run the Docker container:**

    ```bash
    docker run --name transformer-implementation-container --rm --network=host transformer-implementation [runner.device=gpu] [dataset.batch_size=8]
    ```

### Monitoring with TensorBoard
TensorBoard has been integrated for visualizing training progress. To view logs, run:

```bash
tensorboard --logdir=./runs
```
Then, open a browser and go to `http://localhost:6006/`.

## Code Structure
The codebase has been restructured to be more modular and maintainable:

- `config/`: Configuration files managed by Hydra.
- `data/`: Data loading and preprocessing scripts.
- `model/`: Model architecture and related utilities.
- `training/`: Training loop and evaluation logic.
- `utils/`: Utility functions and helpers.

## Known Issues
[List any known issues here]

## Contributions
Feel free to open issues or pull requests for any suggestions or improvements. Bug reports are also welcome.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## References
Attention Is All You Need by Vaswani et al.

Feel free to modify the details to fit your repository!