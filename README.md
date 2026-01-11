Narrative Consistency Analysis System
=======================================

Overview
--------
The Narrative Consistency Analysis System is designed to verify whether character backstories align with the established events of classic literature. By leveraging biologically-inspired neural networks, specifically the Dragon Hatchling (BDH) architecture, the system offers a robust method for detecting subtle narrative contradictions.

Unlike traditional Transformers that require massive datasets, this system utilizes sparse Hebbian learning to efficiently model character relationships and plot progression from limited text.

Key Features
------------
*   **Novel-Specific Pretraining**: Language models are trained on specific literary works to capture their unique style and narrative logic.
*   **Entity Threading**: The system extracts continuous character arcs (threads) rather than random chunks, ensuring the model understands character development over time.
*   **Perplexity Delta Scoring**: Consistency is measured by calculating the reduction in perplexity when the model is conditioned on a given backstory.
*   **Calibrated Classification**: A logistic regression model combines perplexity scores with retrieval metrics to output a probability of consistency.

Architecture and Methodology
----------------------------
### The BDH Model (TextPath)
The core of the system is TextPath, a custom implementation of the Dragon Hatchling architecture. It features:
*   **Sparse Activations**: Only a small fraction of neurons activate for any given input, leading to highly interpretable representations.
*   **Hebbian Learning**: Synaptic weights are updated based on co-activation, allowing the model to "learn" narrative patterns dynamically.
*   **Scale-Free Connectivity**: The network topology mimics biological brain structures for efficient information flow.

### Consistency Detection Pipeline
1.  **Data Ingestion**: The system reads the full text of novels (`The Count of Monte Cristo` and `In Search of the Castaways`).
2.  **Thread Extraction**: All paragraphs related to key characters are aggregated into coherent narrative threads.
3.  **Model Training**: TextPath models are pretrained on both the raw novel text (70%) and the extracted character threads (30%).
4.  **Scoring**: For a given backstory, the system computes the Perplexity Delta ($\Delta$). A positive $\Delta$ indicates consistency, while a negative $\Delta$ suggests a contradiction.
5.  **Prediction**: A calibrated classifier uses the $\Delta$ and retrieval similarity scores to predict the final label (Consistent or Contradictory).

System Requirements
-------------------
*   **Python**: 3.11 or higher
*   **Hardware**: GPU recommended for pretraining (approx. 8GB VRAM), though CPU inference is supported.
*   **Storage**: Sufficient space for model checkpoints (~100MB) and dataset.

Installation
------------
1.  Clone the repository.
2.  Create a virtual environment:
    ```bash
    conda create -n nc_analysis python=3.11
    conda activate nc_analysis
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

Execution
---------
The project is controlled via the `run_pipeline.py` script.

### Standard Workflow
To run the full pipeline (training, evaluation, visualization, and prediction):
`python run_pipeline.py --mode full`

### specific Operations
*   **Pretrain Models**: `python run_pipeline.py --mode pretrain`
*   **Train Calibration**: `python run_pipeline.py --mode train`
*   **Generate Predictions**: `python run_pipeline.py --mode predict`
*   **Evaluate Performance**: `python run_pipeline.py --mode evaluate`
*   **Create Visualizations**: `python run_pipeline.py --mode visualize`

Project Directory Layout
------------------------
*   `src/`: Core source code.
    *   `data_processing/`: Retrieval and entity threading logic.
    *   `models/`: BDH model definitions and training scripts.
    *   `training/`: Calibration and optimization routines.
    *   `analysis/`: Consistency scoring algorithms.
    *   `visualization/`: Plotting utilities.
*   `Dataset/`: Source novels and training/test data.
*   `models/`: Saved model checkpoints and tokenizer files.
*   `visualizations/`: Generated analysis charts and graphs.
*   `outputs/`: Configuration artifacts.

Visualization
-------------
The system produces several plots to aid in analysis:
*   **Delta Distribution**: Visualizes the separation between consistent and contradictory samples.
*   **Confusion Matrix**: Displays classification performance.
*   **Calibration Curve**: Verifies the reliability of predicted probabilities.
*   **Feature Importance**: Shows the contribution of different metrics to the final decision.
