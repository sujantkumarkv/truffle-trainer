import os
import subprocess
import yaml
from prisma import Prisma
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModel
from prisma import Prisma
from hypersearch import HyperSearch

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
    def __init__(self, job_id, mock_job=None, mock_AI=None):
        self.job_id = job_id
        self.base_path = os.path.join(os.getcwd(), str(job_id))
        self.dataset_path = os.path.join(self.base_path, "data", "data.jsonl") # jsonl file with "user" & "assistant" instruct-tune data (as of now)
        self.model_path = os.path.join(self.base_path, "model")
        self.tokenizer_path = os.path.join(self.base_path, "tokenizer")
        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(self.dataset_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)
        self.prisma = PrismaClientSingleton.instance()
        ##### local testing only ####
        self.mock_job = mock_job
        self.mock_AI = mock_AI
        self.prisma.job.find_unique = lambda where: self.mock_job if where['id'] == self.job_id else None
        self.prisma.ai.find_unique = lambda where: self.mock_AI if where['id'] == self.mock_job['baseAiId'] else None
        ##############################

    def getCheckpoint(self):
        job = self.prisma.job.find_unique(where={'id': self.job_id})
        checkpoint = None
        if job and job.baseAiId:
            AIType = job.baseAI.type
            if AIType == "LlamaForCausalLM":
                checkpoint = "meta-llama/Llama-2-7b-hf"
            elif AIType == "MistralForCausalLM":
                checkpoint = "mistralai/Mistral-7B-Instruct-v0.2"
            elif AIType == "PhiForCausalLM":
                checkpoint = "microsoft/phi-1"
            else:
                raise ValueError(f"Checkpoint for your model/tokenizer not found !! Please download yourself manually.")
        
        return checkpoint
            

    async def downloadData(self):
        # Check if the dataset already exists (FOR LOCAL TESTING ONLY)
        if not os.path.exists(self.dataset_path):
            job = self.prisma.job.find_unique(where={'id': self.job_id})
            if job and job.dataset:
                subprocess.run(["gsutil", "cp", job.dataset, dataset_path], check=True)

    async def downloadModel(self):
        # Check if the model already exists (FOR LOCAL TESTING ONLY)
        if not os.path.exists(self.model_path):
            job = self.prisma.job.find_unique(where={'id': self.job_id})
            ai = None
            if job and job.baseAiId:
                ai = self.prisma.ai.find_unique(where={'id': job.baseAiId})

            if ai and ai.weights:
                repo_id, revision = ai.weights.split(':') if ":" in ai.weights else (ai.weights, None)
                snapshot_download(repo_id=repo_id, revision=revision, ignore_patterns="quantized/*", local_dir=self.model_path, 
                                    use_auth_token=True, local_dir_use_symlinks=False, token="")
                # get checkpoint
                tokenizer_checkpoint = self.getCheckpoint()
                # get tokenizer
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint, cache_dir=self.tokenizer_path)

    async def createConfig(self):
        job = self.prisma.job.find_unique(where={'id': self.job_id})
        if job and job.config:
            config_path = os.path.join(self.base_path, 'config.yml')
            
            # Assuming job.config is a string containing YAML-formatted data,
            # first parse it into a Python dictionary
            config_data = yaml.safe_load(job.config)
            
            # HYPERPARAMS SEARCH & then ONLY DUMP THE FINAL config.yml
            # get the final config returned from hypersearch.py
            hypersearchObj = HyperSearch(model_path=self.model_path, dataset_path=self.dataset_path, tokenizer_path=self.tokenizer_path)
            final_config = hypersearchObj.main() 
            # this should wait & only then proceed to dump the final_config yaml, else: will look for async ways.
            with open(config_path, 'w') as file:
                yaml.dump(final_config, file)
