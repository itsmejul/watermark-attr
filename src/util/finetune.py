# Unsloth must be imported before transformers so its patches apply.
from unsloth import FastLanguageModel
from transformers import TrainerCallback
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer
from datasets import Dataset
import torch
import os
import time
from src.util.filereader import write_path_file, load_or_create_path_file, get_lora_adapter_path

UNSLOTH_MAX_SEQ_LENGTH = 512


class SaveAdapterAtEpochCallback(TrainerCallback):
    def __init__(self, save_epochs, experiment_path):
        self.save_epochs = save_epochs
        self.experiment_path = experiment_path

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = round(state.epoch)
        if epoch in self.save_epochs:
            save_path = get_lora_adapter_path(self.experiment_path + [str(epoch)])
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            state.save_to_json(os.path.join(save_path, "trainer_state.json"))
            print(f"Saved LoRA adapter and trainer state at epoch {epoch} to {save_path}")


def finetune(texts, eval_texts, config, experiment_path, add_special_tokens=False, max_length=300):
    """
    Fine-tune a base model with LoRA on the given texts.

    Parameters
    ----------
    texts : list[str]
        Training texts.
    eval_texts : list[str] or None
        Held-out texts for per-epoch eval loss. Pass None to skip eval.
    config : dict
        Must include: epochs, save_every_n_epochs, train_model, target_modules,
        rank, lora_alpha, lora_dropout, bias, learning_rate, batch_size.
    experiment_path : list[str]
        Path components used for adapter and metadata files.
    """
    epochs = config["epochs"]
    step = config["save_every_n_epochs"]
    save_epochs = list(range(step, epochs, step))
    if not save_epochs or save_epochs[-1] != epochs:
        save_epochs.append(epochs)

    callback = SaveAdapterAtEpochCallback(
        save_epochs=save_epochs,
        experiment_path=experiment_path,
    )
    callbacks = [callback]

    start_time = time.time()
    model_name = config["train_model"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=UNSLOTH_MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(samples):
        tokens = tokenizer(
            samples["text"],
            truncation=True,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )
        tokens["length"] = [len(ids) for ids in tokens["input_ids"]]
        return tokens

    train_dataset = Dataset.from_dict({"text": texts}).map(tokenize_function, batched=True)

    eval_dataset = None
    if eval_texts is not None and len(eval_texts) > 0:
        eval_dataset = Dataset.from_dict({"text": eval_texts}).map(tokenize_function, batched=True)
    train_eval_dataset = train_dataset.shuffle(seed=50).select(range(min(200, len(train_dataset))))

    target_modules = config["target_modules"]
    rank = config["rank"]
    lora_alpha = config["lora_alpha"]
    lora_dropout = config["lora_dropout"]
    bias = config["bias"]

    # Unsloth-patched LoRA wrap. Equivalent to peft's get_peft_model + LoraConfig
    # but uses Unsloth's fused kernels and its faster gradient-checkpointing

    model = FastLanguageModel.get_peft_model(
        model,
        r=rank,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        use_gradient_checkpointing=False, # "unsloth "
        random_state=42,
    )
    model.print_trainable_parameters()
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    adapter_save_dir = get_lora_adapter_path(experiment_path)
    os.makedirs(adapter_save_dir, exist_ok=True)

    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]

    training_args = TrainingArguments(
        output_dir=adapter_save_dir,
        save_strategy="no",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        # Constant LR with brief warmup 
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=0.03,
        bf16=True,
        # Log every step to get the loss curve
        logging_steps=1,
        # Held-out eval loss every epoch
        eval_strategy="epoch",
        report_to="none",
        group_by_length=True,
        length_column_name="length",
        dataloader_num_workers=5,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset={"heldout": eval_dataset, "train": train_eval_dataset} if eval_dataset is not None else {"train": train_eval_dataset},
        data_collator=data_collator,
        args=training_args,
        callbacks=callbacks,
    )
    print("Starting training")
    trainer.train()

    trainer.model.save_pretrained(adapter_save_dir)
    # Persist the full log_history (per-step train loss, per-epoch eval loss,
    # grad_norm, lr) 
    trainer.state.save_to_json(os.path.join(adapter_save_dir, "trainer_state.json"))
    print("Saved lora adapter and trainer state")

    end_time = time.time()
    latency = end_time - start_time
    latency_dict = load_or_create_path_file(experiment_path, "latency.json")
    latency_dict["train"] = end_time - start_time
    write_path_file(experiment_path, "latency.json", latency_dict)

    del model, tokenizer, trainer
    torch.cuda.empty_cache()