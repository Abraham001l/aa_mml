# Remote GPU Compute & SLURM Cluster Manual (`alien1` & `alien2`)

> **Target Audience:** Autonomous AI Agents, LLM Researchers, and Engineers running high-performance GPU model training, GNNs, and hyperparameter sweeps for the American Airlines Forecasting project.  
> **Hosts:** `alien1`, `alien2` (Alienware Workstations with NVIDIA GPUs & SLURM Workload Manager)  
> **User Account:** `abraham01cyb`  
> **Password:** `Cr3#!0Ne8cyb`  
> **Python Command:** `python3`

---

## 1. Quick Connection Reference

### Direct SSH
```bash
# Connect to Node 1
ssh abraham01cyb@alien1

# Connect to Node 2
ssh abraham01cyb@alien2
```

### Passwordless SSH Key Setup (Highly Recommended for LLM Agents)
To prevent agents or automated scripts from hanging on interactive password prompts, set up SSH keys once from your local machine:
```bash
# Generate key if you don't have one:
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519

# Copy public key to both nodes (enter password 'Cr3#!0Ne8cyb' when prompted):
ssh-copy-id abraham01cyb@alien1
ssh-copy-id abraham01cyb@alien2
```
*Once copied, all SSH and SCP commands will execute non-interactively without prompting for passwords.*

---

## 2. Directory Isolation Standard (Clean Workspace Policy)

To ensure the host machine stays clean and user files are not cluttered, **every agent must create and work exclusively inside its own disposable directory**:

```bash
# Standard directory format:
~/scratch_<agent_name>_aa_mml/

# Examples:
mkdir -p ~/scratch_antigravity_aa_mml/
mkdir -p ~/scratch_gpt_agent_aa_mml/
```

> **Cleanup Rule:** When your training runs are finished and outputs are synced back, the user or agent can cleanly delete the entire workspace in one command:
> ```bash
> rm -rf ~/scratch_*_aa_mml/
> ```

---

## 3. Remote Environment Setup & GPU Verification

### 3.1 Check NVIDIA Hardware & Driver
```bash
nvidia-smi
```

### 3.2 Initialize Isolated Python Virtual Environment
```bash
# Navigate to your isolated scratch folder:
cd ~/scratch_antigravity_aa_mml/

# Create and activate a dedicated virtual environment:
python3 -m venv venv
source venv/bin/activate

# Upgrade pip:
pip install --upgrade pip

# Install PyTorch with CUDA support:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Project Data & Modeling Stack:
pip install xgboost lightgbm catboost scikit-learn pandas pyarrow scipy networkx matplotlib
```

### 3.3 Verify CUDA Acceleration in Python
```bash
python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 4. SLURM Workload Management Guide

Both nodes have a SLURM cluster manager installed. Use SLURM to submit heavy training jobs, multi-GPU scripts, and long-running sweeps.

### 4.1 Common SLURM CLI Commands
| Command | Purpose | Example |
| :--- | :--- | :--- |
| `sinfo` | View cluster partition state and node availability | `sinfo` |
| `squeue` | View running and pending jobs in the queue | `squeue -u abraham01cyb` |
| `sbatch <script.slurm>` | Submit an asynchronous batch job | `sbatch train_gnn.slurm` |
| `scancel <job_id>` | Terminate a running or queued job | `scancel 1234` |
| `srun --pty bash` | Launch an interactive shell on an allocated GPU node | `srun --gres=gpu:1 --pty bash` |

---

### 4.2 Production SLURM Batch Script Template (`submit_gpu.slurm`)

Save this template inside your remote scratch folder:

```bash
#!/bin/bash
#SBATCH --job-name=aa_gpu_train
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00

# 1. Echo job metadata
echo "Starting SLURM Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

# 2. Activate Python environment
source ~/scratch_antigravity_aa_mml/venv/bin/activate

# 3. Navigate to workspace
cd ~/scratch_antigravity_aa_mml/code/

# 4. Execute GPU training script
python3 -u train_spatiotemporal_gnn.py \
    --data_dir ../data \
    --output_dir ../outputs \
    --epochs 100 \
    --batch_size 128 \
    --lr 0.001

echo "Job completed at $(date)"
```

Submit with:
```bash
mkdir -p logs
sbatch submit_gpu.slurm
```

---

### 4.3 Interactive GPU Debugging Session
If you need to test code, check memory consumption, or debug a script interactively with a live GPU:
```bash
srun --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=01:00:00 --pty bash
```

---

## 5. Data & Artifact Transfer (Syncing Local $\leftrightarrow$ Remote)

### 5.1 Push Code & Data to Remote Cluster (From Local Machine)
Run from the root of your local repository (`c:\Users\abrah\Desktop\projects\aa_mml\`):

```powershell
# 1. Create target directories on remote:
ssh abraham01cyb@alien1 "mkdir -p ~/scratch_antigravity_aa_mml/data ~/scratch_antigravity_aa_mml/code"

# 2. Sync read-only input dataset:
scp ./abraham/global_data/airport_info.csv abraham01cyb@alien1:~/scratch_antigravity_aa_mml/data/
scp ./abraham/global_data/pax_forecasting.csv abraham01cyb@alien1:~/scratch_antigravity_aa_mml/data/

# 3. Sync external demographic and migration tables:
scp ./abraham/agent_workspace/data_external/*.parquet abraham01cyb@alien1:~/scratch_antigravity_aa_mml/data/

# 4. Sync your modeling scripts:
scp -r ./abraham/agent_workspace/22_deep_spatial_gravity/* abraham01cyb@alien1:~/scratch_antigravity_aa_mml/code/
```

### 5.2 Pull Trained Models & Predictions Back to Local
Once the remote GPU job finishes:
```powershell
# Copy back backtest metrics and predictions:
scp abraham01cyb@alien1:~/scratch_antigravity_aa_mml/outputs/backtest_metrics.json ./abraham/agent_workspace/25_gpu_model/outputs/
scp abraham01cyb@alien1:~/scratch_antigravity_aa_mml/outputs/backtest_predictions.parquet ./abraham/agent_workspace/25_gpu_model/outputs/
```

---

## 6. Guidelines for Autonomous AI Agents

1. **Non-Interactive Execution:** Agents must pass commands directly over SSH rather than opening an interactive bash shell that hangs waiting for input:
   ```bash
   ssh abraham01cyb@alien1 "bash -c 'source ~/scratch_*/venv/bin/activate && python3 -u script.py'"
   ```
2. **Background Detachment (`nohup` / `sbatch`):** Always submit training via `sbatch` or `nohup ... > run.log 2>&1 &` so that SSH disconnects do not terminate long model runs.
3. **Structured Outputs:** Always save final scores as `backtest_metrics.json` matching the local schema (`shared_eval/metrics.py`) so results can be evaluated and logged directly to `experiments.md`.
4. **Cleanup:** Always remove temporary checkpoints (`rm -rf *.pt.tmp`) and notify the user when the scratch folder is ready for deletion.
