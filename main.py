import asyncio
from utilities import DataManager, PrismaClientSingleton
from prisma import Prisma
from trainer import Trainer  # Assuming the Trainer class is defined in trainer.py

async def process_job(job_id):
    data_manager = DataManager(job_id)
    # Perform initial steps in parallel
    await asyncio.gather(
        # data_manager.downloadData(),
        # data_manager.createConfig(),
        # data_manager.downloadModel()
    )
    # Initiate the training process
    trainer = Trainer(job_id)
    await trainer.train()

async def main():
    prisma = PrismaClientSingleton.instance()
    jobs = prisma.job.find_many(where={'status': 'created'})

    tasks = [process_job(job.id) for job in jobs]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())


# First you want to retrieve an array of jobs available in a queue
# For each job, first process it by: 1. Downloading the dataset locally, 2. Downloading the model locally, 3. Saving the YAML, 4. Running the training script until exit or failure
# 5. Saving logs into GCP if fail or success, 6. If success, saving the newly trained model. If its a Base fork, create a repo for the user in HF, and make a commit with "weights"
# Quantized in huggingface and push, else just get the users repo in HF, and make a commit and push to existing repo if its not a base model
# Then we can exit, say job is complete and go onto the next job