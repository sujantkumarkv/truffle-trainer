import asyncio
from utilities import DataManager, PrismaClientSingleton
from prisma import Prisma
from trainer import Trainer
from enums import ChatConfig, AI, AIType, JobStatus

async def process_job(job_id, mock_job, mock_AI):
    data_manager = DataManager(job_id, mock_job=mock_job, mock_AI=mock_AI)
    data_manager.prisma.job.find_unique = lambda where: job_data

    # print(f"base bath: {data_manager.base_path}")
    # perform initial steps in parallel
    await asyncio.gather(
        data_manager.downloadData(),
        data_manager.downloadModel(),
        data_manager.createConfig(),
    )
    # # Initiate training with the best hyperparams config
    # trainer = Trainer(job_id)
    # await trainer.train()

async def main():
    # mock based on prisma schema
    mock_AI = {
        'id': 'base_ai_id',
        'createdAt': '2023-01-01T00:00:00Z',
        'weights': 'sample_job_id/model/',
        'context': 2048,
        'size': 1.5,
        'type': AIType.PhiForCausalLM,
        'baseForJobs': [],
    }

    mock_job = {
        'id': 'sample_job_id',
        'status': 'created',
        'config': 'mock_job_id/ft_config.yml',  # yaml string
        'dataset': 'mock_job_id/data/data.jsonl',
        'baseAiId': 'base_ai_id',
        'cc': ChatConfig.phi_2,
        'baseAI': mock_AI,
    }
    await process_job('mock_job_id', mock_job=mock_job, mock_AI=mock_AI)

if __name__ == "__main__":
    asyncio.run(main())