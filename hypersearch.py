import time
import tempfile
import torch
import ray
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.utils.data import Dataset, TensorDataset, Subset
from torch.utils.data import DataLoader
from peft import  LoraConfig, TaskType, get_peft_model
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune import  with_parameters, with_resources, TuneConfig
from ray.tune.tuner import Tuner
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score

"""
the dataset format is currently jsonl (instruct finetune). we can use `model = get_peft_model` & train with for loop & `loss.backward()`
but, since hyperparam-search is also many finetune steps with diff configs, we can use axolotl only AND 
with subset of data prolly (condition: how much is data? to then take a subset for hyperparam-search or take the full data then)
"""

def create_data_loaders(dataset, batch_size, sample_pct, collate_fn):
    subset_size = int(len(dataset) * sample_pct)
    
    indices = torch.randperm(len(dataset))[:subset_size]
    subset = Subset(dataset, indices)
    
    data_loader = DataLoader(subset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
    
    return data_loader


def compute_metrics(predicted, actual):
    predicted = np.argmax(predicted.detach().numpy(), axis=1)
    f1 = f1_score(actual, predicted, average='weighted')
    precision = precision_score(actual, predicted, average='weighted')
    recall = recall_score(actual, predicted, average='weighted')
    accuracy = accuracy_score(actual, predicted)
    return f1, precision, recall, accuracy

def train_model(config, model, train_data, val_data, sample_pct, batch_size):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(f"runs at {timestamp}")
    best_eval_loss = float('inf')
    NUM_EPOCHS = 2 
    alpha = config["r"]

    lora_config = LoraConfig(
        r=config["r"], 
        lora_alpha=alpha,
        target_modules=config["target_modules"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type=TaskType.SEQ_CLS)

    model = get_peft_model(model, lora_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    train_loader = create_data_loaders(train_data, batch_size=batch_size, sample_pct=sample_pct, collate_fn=data_collator)
    val_loader = create_data_loaders(val_data, batch_size=batch_size, sample_pct=sample_pct, collate_fn=data_collator)

    if hasattr(train, "get_checkpoint") and train.get_checkpoint():
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
        running_eval_loss = 0
        running_eval_f1 = 0
        running_eval_auc = 0
        running_eval_precision = 0
        running_eval_recall = 0
        running_eval_accuracy = 0

        for i, batch in enumerate(val_loader):
            output = model(**batch)
            running_eval_loss += output.loss
            f1, precision, recall, accuracy = compute_metrics(output.logits, batch['labels'])
            running_eval_f1 += f1
            running_eval_precision += precision
            running_eval_recall += recall
            running_eval_accuracy += accuracy

        avg_eval_loss = running_eval_loss / len(val_loader)
        avg_eval_f1 = running_eval_f1 / len(val_loader)
        avg_eval_precision = running_eval_precision / len(val_loader)
        avg_eval_recall = running_eval_recall / len(val_loader)
        avg_eval_accuracy = running_eval_accuracy / len(val_loader)

        print(f"Avg validation loss ==>: {float(avg_eval_loss)}, F1 Score ==> {float(avg_eval_f1)}, Precision ==> {float(avg_eval_precision)}, Recall ==> {float(avg_eval_recall)}, Accuracy ===> {float(avg_eval_accuracy)}")
        writer.add_scalar("Eval Loss", avg_eval_loss, epoch + 1)
        writer.add_scalar("F1 Score", avg_eval_f1, epoch + 1)
        writer.add_scalar("Precision", avg_eval_precision, epoch + 1)
        writer.add_scalar("Recall", avg_eval_recall, epoch + 1)
        writer.add_scalar("Accuracy", avg_eval_accuracy, epoch + 1)
        writer.flush()

        train.report({"loss": avg_eval_loss.item()})

        print("Finished Training")


def create_search_space():
    return {
        "lr": tune.loguniform(1e-4, 1e-1),
        "r": tune.choice([2, 4, 6, 8, 10, 16]), 
        # "target_modules": tune.choice([["q_lin"], ["v_lin"], ["q_lin", "v_lin"]]), 
        "lora_dropout": tune.uniform(0.1, 0.5), 
    }


def main(data, sample_pct=0.5, batch_size, max_num_epochs=10, num_samples=5):
    model = data[0]
    train_data = data[1]
    val_data = data[2]

    config = create_search_space()

    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)

    tuner = Tuner(
        with_resources(
            tune.with_parameters(train_model, model=model, train_data=train_data, 
                            val_data=val_data, sample_pct=sample_pct, batch_size=batch_size),
            resources={"cpu": 2}),
        tune_config=TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
        
    )
    results = tuner.fit()
    
    best_result = results.get_best_result("loss", "min", filter_nan_and_inf=True)

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(best_result.metrics["loss"]))

# download the model, config & dataset is in `truffle-trainer/utitities.py`
train_data_ray = ray.put(train_data_tokenized)
val_data_ray = ray.put(val_data_tokenized)
model_ray = ray.put(model)
main((model, train_data_ray, val_data_ray))