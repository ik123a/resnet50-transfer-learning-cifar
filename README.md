# ResNet-50 Transfer Learning on CIFAR-10 / CIFAR-100

A complete, optimized **PyTorch** implementation utilizing a pretrained **ResNet-50** architecture for image classification on the **CIFAR-10** (or CIFAR-100) dataset. 

This repository compares three distinct neural network freezing strategies for transfer learning, implements validation early stopping, learning rate hyperparameter tuning, confusion matrices, classification reports, and test inference.

---

## 🚀 Key Features

* **Three Freezing Strategies Compared:**
  * **Fully Frozen:** Only the final classifier header (FC layer) is trainable.
  * **Partially Frozen:** The early convolution blocks are frozen while the deep residual blocks (`layer3` and `layer4`) and the FC header are fine-tuned.
  * **Fully Trainable:** The entire ResNet-50 network is trainable to fully adapt to the CIFAR distribution.
* **Stable Batch Normalization:** Explicitly keeps frozen layer BatchNorm blocks in evaluation mode during training to maintain stable pre-trained ImageNet running statistics.
* **Hyperparameter Tuning:** Automated learning rate candidate search evaluating the best checkpointed epoch validation accuracy.
* **Early Stopping & Checkpoint Saving:** Monitors validation loss, saving the best-performing weights and terminating training early if validation loss plateaus.
* **Robust Hardware & Environment Fallbacks:**
  * **Windows Multiprocessing Support:** Dynamically handles `NUM_WORKERS` depending on the operating system (`0` on Windows to prevent multiprocessing hangs, `2` on Colab/Linux).
  * **Flexible Progress Bars:** Automatically detects whether it is running inside an interactive notebook (using `tqdm.notebook` for inline HTML bars) or a standard terminal (using text `tqdm` progress bars).

---

## 📁 Repository Structure

* `resnet50_transfer_learning_cifar_colab.ipynb` - The primary PyTorch Jupyter notebook (Google Colab ready).
* `.gitignore` - Standard configuration ignoring dataset directories and weight checkpoints to keep the repository lightweight.

---

## 🛠️ Getting Started

### Google Colab (Recommended)
1. Upload the `.ipynb` file to your Google Drive or open it directly from your GitHub account.
2. Under **Runtime** > **Change runtime type**, select **T4 GPU** (or any available GPU accelerator).
3. Run all cells. The notebook will automatically set `FAST_RUN = False` to execute the full training run in under 45 minutes on a GPU.

### Local Execution (CPU / GPU)
1. Clone the repository:
   ```bash
   git clone https://github.com/ik123a/resnet50-transfer-learning-cifar.git
   cd resnet50-transfer-learning-cifar
   ```
2. Install the dependencies:
   ```bash
   pip install torch torchvision seaborn scikit-learn tqdm pandas matplotlib
   ```
3. Open the notebook in Jupyter Notebook or VS Code, or convert it to a script and run:
   ```bash
   python resnet50_transfer_learning_cifar_colab.ipynb
   ```
   *Note: If no CUDA GPU is detected, the script will automatically activate `FAST_RUN = True` mode, executing a lightweight dry-run to verify the pipeline in minutes.*

---

## 📊 Results Summary

The experiments evaluate test accuracy across the three strategies:
* **Fully Frozen:** Fastest to execute, useful as a solid baseline.
* **Partially Frozen:** Strong balance between computation time and accuracy.
* **Fully Trainable:** Highest flexibility, achieving the highest final accuracy at the cost of longer computation time.

---

## 💳 License
This project is open-source and available under the [MIT License](LICENSE).
