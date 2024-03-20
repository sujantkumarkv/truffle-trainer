import os
import subprocess
import yaml
from prisma import Prisma
from huggingface_hub import snapshot_download


from prisma import Prisma

class PrismaClientSingleton:
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = Prisma()
            cls._instance.connect()
        return cls._instance

    @classmethod
    def disconnect(cls):
        if cls._instance is not None:
            cls._instance.disconnect()
            cls._instance = None

class DataManager:
    def __init__(self, job_id):
        self.job_id = job_id
        self.base_path = os.path.join(os.getcwd(), str(job_id))
        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(os.path.join(self.base_path, "data"), exist_ok=True)
        os.makedirs(os.path.join(self.base_path, "model"), exist_ok=True)
        self.prisma = PrismaClientSingleton.instance()


    async def createConfig(self):
        job = self.prisma.job.find_unique(where={'id': self.job_id})
        if job and job.config:
            config_path = os.path.join(self.base_path, 'config.yml')
            
            # Assuming job.config is a string containing YAML-formatted data,
            # first parse it into a Python dictionary
            config_data = yaml.safe_load(job.config)
            
            # Then, dump the dictionary into a YAML file
            with open(config_path, 'w') as file:
                yaml.dump(config_data, file)

    async def downloadData(self):
        job = self.prisma.job.find_unique(where={'id': self.job_id})
        if job and job.dataset:
            dataset_path = os.path.join(self.base_path, "data", "data.jsonl")
            subprocess.run(["gsutil", "cp", job.dataset, dataset_path], check=True)

    async def downloadModel(self):
        job = self.prisma.job.find_unique(where={'id': self.job_id})
        ai = None
        if job and job.baseAiId:
            ai = self.prisma.ai.find_unique(where={'id': job.baseAiId})

        if ai and ai.weights:
            repo_id, revision = ai.weights.split(':') if ":" in ai.weights else (ai.weights, None)
            model_path = os.path.join(self.base_path, "model")
            snapshot_download(repo_id=repo_id, revision=revision, ignore_patterns="quantized/*", local_dir=model_path, use_auth_token=True, local_dir_use_symlinks=False, token="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")
