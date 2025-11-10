import argparse
import subprocess
import sys
import os
from pathlib import Path

def create_slurm_script(
    command, 
    job_name, 
    partition, 
    time, 
    cpus, 
    mem, 
    gpus, 
    constraint,
    env_name,
    email,
    node_type
):
    """Dynamically create SLURM sbatch script content based on the provided parameters"""
    
    # If user didn't provide job_name, generate one automatically from command
    if not job_name:
        job_name = f"{command.replace(' ', '_')}"

    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)

    # Build GPU-related configuration based on node type
    gpu_config = ""
    if node_type == "gpu":
        gpu_config = f"#SBATCH --gres=gpu:{gpus}"
        if constraint:
            gpu_config += f'\n#SBATCH --constraint="{constraint}"'

    # Build script template using f-string
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --time={time}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
{gpu_config}
#SBATCH --output=logs/%j_{job_name}.out
#SBATCH --error=logs/%j_{job_name}.err
#SBATCH --mail-type=BEGIN,END,FAIL  # Email notifications (on job start, end, or fail)
#SBATCH --mail-user={email}  # Email address for notifications

# Change to working directory
cd {os.getcwd()}

echo "=========================================================="
echo "Starting on $(hostname)"
echo "Time is $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Node type: {node_type.upper()}"
echo "Running command: {command}"
echo "=========================================================="

# Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate {env_name}

# Verify environment activation
echo "Python path: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Set just runtime directory to avoid permission errors
# Try user's home directory first, fallback to project directory
if [ -n "$HOME" ] && [ -w "$HOME" ]; then
    export JUST_RUNTIME_DIR="$HOME/.just_runtime"
else
    export JUST_RUNTIME_DIR="{os.getcwd()}/.just_runtime"
fi
mkdir -p "$JUST_RUNTIME_DIR" 2>/dev/null || true
chmod 700 "$JUST_RUNTIME_DIR" 2>/dev/null || true
echo "JUST_RUNTIME_DIR set to: $JUST_RUNTIME_DIR"

# Run the script
{command}

# Send discord notification after job completion
if command -v discord &> /dev/null; then
    discord "command '{command}' finished on job $SLURM_JOB_ID"
else
    echo "Discord command not found, skipping notification"
fi
"""
    return slurm_script

def submit_job(args):
    """Submit SLURM job"""
    
    # 1. Create SLURM script content
    script_content = create_slurm_script(
        command=args.command,
        job_name=args.job_name,
        partition=args.partition,
        time=args.time,
        cpus=args.cpus,
        mem=args.mem,
        gpus=args.gpus,
        constraint=args.constraint,
        env_name=args.env_name,
        email=args.email,
        node_type=args.node_type
    )
    
    print("--- SLURM script to be submitted ---")
    print(script_content)
    print("-----------------------------------")
    
    try:
        # 2. Pass script content to sbatch via stdin and execute
        result = subprocess.run(
            ['sbatch'],
            input=script_content,
            capture_output=True,
            text=True,
            check=True
        )
        print("✅ Job submitted successfully!")
        print(f"   {result.stdout.strip()}")

    except FileNotFoundError:
        print("❌ Error: 'sbatch' command not found. Please ensure Slurm environment is properly configured.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print("❌ Error: Job submission failed.", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Submit python command as SLURM job.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show default values
    )
    
    # --- Required arguments ---
    parser.add_argument('--command', help='Python command to run (e.g., train_model)')
    
    # --- Optional arguments (override SLURM default configuration) ---
    parser.add_argument('--node_type', default='gpu', choices=['gpu', 'cpu'], 
                        help='Node type: gpu (use GPU node) or cpu (use CPU node)')
    parser.add_argument('--job_name', '-n', help='Job name (default: auto-generated from command)')
    parser.add_argument('--time', '-t', default='120:00:00', help='Job runtime (D-HH:MM:SS)')
    parser.add_argument('--partition', '-p', default='long', help='Partition to submit to')
    parser.add_argument('--cpus', '-c', type=int, default=32, help='Number of CPU cores per task')
    parser.add_argument('--mem', '-m', default='240G', help='Memory size to request (e.g., 240G)')
    parser.add_argument('--gpus', '-g', type=int, default=1, help='Number of GPUs to request (only effective when node_type=gpu)')
    parser.add_argument('--constraint', default='', help='GPU type constraint (e.g., "v100|a100", only effective when node_type=gpu)')
    parser.add_argument('--env_name', '-e', default='mitoem2', help='Conda environment to activate')
    parser.add_argument('--email',default='liupen@bc.edu', help='Email address for notifications')
    args = parser.parse_args()
    

    submit_job(args)

if __name__ == '__main__':
    main()