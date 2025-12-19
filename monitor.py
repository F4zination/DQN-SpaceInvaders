import subprocess
import time
import sys

def run_training():
    max_restarts = 5
    restart_count = 0
    script_to_run = "space_invaders_long_run.py" # Ensure this matches your filename

    while restart_count <= max_restarts:
        print(f"\n{'='*40}")
        print(f"STARTING TRAINING (Attempt {restart_count + 1}/{max_restarts + 1})")
        print(f"{'='*40}\n")

        # Start the training process
        # We use sys.executable to ensure it uses the same python environment
        process = subprocess.Popen(
            [
                sys.executable,
                script_to_run,
                "--try",
                str(restart_count),
            ]
        )
        
        # Wait for the process to finish
        return_code = process.wait()

        if return_code == 0:
            print("Training finished successfully!")
            break
        else:
            restart_count += 1
            if restart_count <= max_restarts:
                wait_time = 30 # Wait 30 seconds before restarting
                print(f"ERROR: Training crashed with code {return_code}.")
                print(f"Restarting in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("FATAL: Maximum restart attempts reached. Training aborted.")

if __name__ == "__main__":
    run_training()
