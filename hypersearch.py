import time
import tempfile
import torch
import ray
import os, json, subprocess
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.utils.data import Dataset, TensorDataset, Subset
from torch.utils.data import DataLoader, Subset
from peft import  LoraConfig, TaskType, get_peft_model
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune import  with_parameters, with_resources, TuneConfig
from ray.tune.tuner import Tuner
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel

"""
the dataset format is currently jsonl (instruct finetune). we can use `model = get_peft_model` & train with for loop & `loss.backward()`
but, since hyperparam-search is also many finetune steps with diff configs, we can use axolotl only AND 
with subset of data prolly (condition: how much is data? to then take a subset for hyperparam-search or take the full data then)
"""

### TODOs (2 hrs max. write to srikanth before 12)
# how/why to use train/val loop
# `Tuner` variables

## later
# upload model logic

class HyperSearch:
    def __init__(self, model_path, dataset_path, tokenizer_path):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.tokenizer_path = tokenizer_path

    def get_split_data(self, val_pct):
        # TODO: look for more heuristics/checks on how to split based on dataset sizes
        # Read .jsonl and parse it
        with open(self.dataset_path, 'r') as file:
            data = [json.loads(line) for line in file]

        # Split into training and validation sets
        train_data, val_data = train_test_split(data, test_size=val_pct)

        return train_data, val_data

    def collate_fn(self, batch):
        # Assuming each element in 'batch' is a dict with 'user' and 'assistant' keys
        batch_text = [each['user'] + '\n' + each['assistant'] + '\n' for each in batch]
        # TODO: try this works like tokenizer(text)['input_ids']  & check the params
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        # TODO: look into this
        # tokenizer(batch_text, padding=True, truncation=True)['input_ids']
        # useful to have `padding` for making batches BUT an issue i faced: tokenizer maynot have PAD token.
        return tokenizer(batch_text, truncation=True)['input_ids']

    def create_data_loaders(dataset, batch_size, sample_pct):
        subset_size = int(len(dataset) * sample_pct)
        
        indices = torch.randperm(len(dataset))[: subset_size]
        subset = Subset(dataset, indices)
        # more on `collate_fn`: https://stackoverflow.com/questions/65279115/how-to-use-collate-fn-with-dataloaders
        data_loader = DataLoader(subset, collate_fn=self.collate_fn, batch_size=batch_size, shuffle=True) 
        return data_loader


    def compute_benchmarks(self, epoch):
        # TODO: appropriate benchmarks
        """
        the heuristics of benchmarks & choosing best hyperparams config based on that is non-trivial;
        because the metrics change based on task too, like user may finetune for classfication & then we may need precision, recall 
        but for most Language, math, coding tasks, different metrics wld be needed & the resulting model wld perform based
        on data & intent of result by user.

        idea: 
        maybe a good choice is to ask user whats' the intent of finetune? 
        what kind of results are expected & what's the nature of data?

        # currently i use eleutherai's harness to get benchmarks..
        # though benchmarks are mostly d*ck-measuring contests now, but that's what's for now.
        """
        # the output directory shall exist
        output_dir = f"hypersearch_eval/epoch_{epoch}"
        os.makedirs(output_dir, exist_ok=True)
        # Construct with dynamic values
        command = [
            "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={self.model_path}",
            "--tasks", "lambada_openai,hellaswag", # more available
            "--device", "cuda:0",
            "--batch_size", "auto",
            "--output_path", output_dir
        ]

        # Run
        result = subprocess.run(command, capture_output=True, text=True)

        # Check if successful
        if result.returncode != 0:
            print(f"Error running command: {result.stderr}")
            return None

        print(f"Command output: {result.stdout}")

    def train_model(config, model, train_data, val_data, sample_pct, batch_size):
        # TODO:  try getting wandb work OR see if we can use tensorboard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        writer = SummaryWriter(f"runs at {timestamp}")
        best_eval_loss = float('inf')
        NUM_EPOCHS = 3
        alpha = config["r"]

        lora_config = LoraConfig(
            r=config["r"], 
            lora_alpha=alpha,
            # target_modules=config["target_modules"],
            lora_dropout=config["lora_dropout"],
            bias="none",
            task_type=TaskType.SEQ_CLS
        )

        model = get_peft_model(model, lora_config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        train_loader = create_data_loaders(train_data, batch_size=batch_size, sample_pct=sample_pct)
        val_loader = create_data_loaders(val_data, batch_size=batch_size, sample_pct=sample_pct)

        
        if hasattr(train, "get_checkpoint") and train.get_checkpoint():
            """
            - checks if the get_checkpoint exists in ray.train, if yes: returns a checkpoint object & loads the checkpoint.
            its expected to be a directory that contains the saved model state and optimizer state. 
            - torch.load() is used to load it. the model's and optimizer's states are restored using the load_state_dict method.
            - If no checkpoint is found, it sets epoch_start to 0, indicating that training should start from the beginning. 
            """
            loaded_checkpoint = train.get_checkpoint()
            with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
                model_state, optimizer_state, epoch_start = torch.load(
                    os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
                )
                model.load_state_dict(model_state)
                optimizer.load_state_dict(optimizer_state)
        else:
            epoch_start = 0

        for epoch in range(epoch_start, NUM_EPOCHS):
            print(f"Epoch {epoch + 1}")
            model.train()
            running_loss = 0
            # train loop
            for j, batch in enumerate(train_loader):
                output = model(**batch)
                loss = output.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                running_loss += loss.item()

                if j % 10 == 9:
                    last_loss = running_loss / j + 1
                    print(f"loss at batch {j + 1} = {last_loss}")
                    tb_x = epoch * len(train_loader) + j + 1
                    writer.add_scalar("loss", last_loss, tb_x)
                    running_loss = 0

            model.eval()
            # val loop
            for i, batch in enumerate(val_loader):
                output = model(**batch)
                running_eval_loss += output.loss
            
            # benchmarks
            self.compute_benchmarks()
            print("benchmarks available in hypersearch_eval/.")
            avg_eval_loss = running_eval_loss / len(val_loader)

            print(f"Avg validation loss =>: {float(avg_eval_loss)}")
            writer.add_scalar("Eval Loss", avg_eval_loss, epoch + 1)
            writer.flush()

            train.report({"loss": avg_eval_loss.item()})

            print("Finished...")


    def create_search_space():
        return {
            "lr": tune.loguniform(1e-4, 1e-1),
            "r": tune.choice([2, 4, 6, 8, 10, 16]), 
            # "target_modules": tune.choice([["q_lin"], ["v_lin"], ["q_lin", "v_lin"]]), 
            "lora_dropout": tune.uniform(0.1, 0.5), 
        }

    # TODO: find more optimal or learn these params too. eg, `batch_size` 
    def main(sample_pct=0.5, batch_size=16, max_num_epochs=10, num_samples=5):
        model = AutoModel.from_pretrained(self.model_path, local_files_only=True) # model is already downloaed in utitlities.py's `downloadModel`
        train_data, val_data = self.get_split_data(val_pct=0.2)

        train_data_ray = ray.put(train_data)
        val_data_ray = ray.put(val_data)
        model_ray = ray.put(model)

        search_space = create_search_space()
        scheduler = ASHAScheduler(
            max_t=max_num_epochs,
            grace_period=1,
            reduction_factor=2)

        tuner = Tuner(
            with_resources(
                tune.with_parameters(train_model, model=model_ray, train_data=train_data_ray, 
                                val_data=val_data_ray, sample_pct=sample_pct, batch_size=batch_size),
                resources={"cpu": 2}), # TODO: decide the values here OR setup dynamic config
            tune_config=TuneConfig(
                metric="loss", # look for more heuristics here
                mode="min",
                scheduler=scheduler,
                num_samples=num_samples,
            ),
            param_space=search_space,
            
        )
        results = tuner.fit()
        
        best_result = results.get_best_result("loss", "min", filter_nan_and_inf=True)

        print("Best trial config: {}".format(best_result.config)) # `get_best_result` returns `Result` object which has `.config` property.
        print("Best trial final validation loss: {}".format(best_result.metrics["loss"]))
        # get the final best config
        best_config = best_result.config
        # Update the original config with the best config values
        final_config.update(best_config)

        return final_config

