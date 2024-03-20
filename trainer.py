import asyncio
import subprocess
import os

class Trainer:
    def __init__(self, job_id):
        self.job_id = job_id

    async def train(self):
        config_path = f"./{self.job_id}/config.yml"
        log_path = f"./{self.job_id}/logs/training.log"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # Start the training process
        with open(log_path, "w") as log_file:
            process = await asyncio.create_subprocess_shell(
                f"accelerate launch -m axolotl.cli.train {config_path}",
                stdout=log_file,
                stderr=log_file
            )

            # Wait for the training process to complete
            await process.wait()

            # Check if the process exited successfully
            if process.returncode == 0:
                print(f"Training for job {self.job_id} completed successfully.")
            else:
                print(f"Training for job {self.job_id} failed with return code {process.returncode}.")

    async def cleanup(self):
        # Placeholder for cleanup logic
        pass