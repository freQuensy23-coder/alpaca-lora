import os
import sys
import warnings
from typing import List

import datasets
import fire
import pandas as pd
import torch
import transformers
import wandb as wandb
from datasets import load_dataset, Dataset

from WandbTrainer import WandbTrainer

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer, EarlyStoppingCallback

from utils.prompter import Prompter


def train(
        # model/data params
        base_model: str = "openlm-research/open_llama_3b",
        data_path: str = "",
        output_dir: str = "./lora-alpaca",
        # training hyperparams
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        cutoff_len: int = 2048,
        # lora hyperparams
        lora_r: int = 4,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = [
            "q_proj",
            "v_proj",
        ],
        # llm hyperparams
        train_on_inputs: bool = False,  # if False, masks out inputs in loss
        add_eos_token: bool = False,
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
        use_custom_prompt: bool = False,
        auto_wandb: bool = False,
        optim: str = "adamw_torch",
        early_stopping: bool = False,
):
    global SophiaG
    if auto_wandb:
        wandb_project = f"{base_model}".replace("/", "-")
        wandb_run_name = f"{wandb_project}-{lora_r}-{lora_alpha}-{lora_dropout}"
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
            f"use_custom_prompt: {use_custom_prompt}\n"
            f"auto_wandb: {auto_wandb}\n"
            f"optim: {optim}\n"
            f"early_stopping: {early_stopping}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference
    if tokenizer.pad_token is None:
        print(
            f"[WARNING] Tokenizer has no pad_token set, setting it to 1. Hardcode other value if your model uses it as padding.")
        tokenizer.pad_token_id = 1

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        if use_custom_prompt:
            full_prompt = data_point["prompt"] + " " + data_point["hypothesis"]
        else:
            full_prompt = prompter.generate_prompt(
                data_point["instruction"],
                data_point["input"],
                data_point["output"],
            )

        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            if not use_custom_prompt:
                user_prompt = prompter.generate_prompt(
                    data_point["instruction"], data_point["input"]
                )
            else:
                user_prompt = data["prompt"]
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            # Masking
            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):  #
        raise NotImplementedError("JSON loading not implemented yet")
    else:
        train_df_parts = [  # TODO REFACTOR!
            datasets.load_dataset(
                "tq-still/still-llm-labeling",
                data_files="still_ds_labeling_v1.json.gz",
                use_auth_token=True,
                split='train',
            ).to_pandas(),
            datasets.load_dataset(
                "tq-still/still-llm-labeling",
                data_files="still_ds_labeling_v2.json.gz",
                use_auth_token=True,
                split='train',
            ).to_pandas(),
            datasets.load_dataset(
                "tq-still/still-llm-labeling",
                data_files="still_ds_labeling_v3.json.gz",
                use_auth_token=True,
                split='train',
            ).to_pandas(),
            datasets.load_dataset(
                "tq-still/still-llm-labeling",
                data_files="still_ds_labeling_v4.json.gz",
                use_auth_token=True,
                split='train',
            ).to_pandas(),
        ]
        train_df = pd.concat(train_df_parts, ignore_index=True)

        train_df = train_df.groupby('q_id', group_keys=False).apply(
            lambda q_df: q_df.nlargest(2, 'mean_rating')
        ).drop_duplicates(subset=['q_id', 'hypothesis']).reset_index(drop=True)

        relevant_columns = ['dialog_id', 'q_id', 'hypothesis', 'context']

        train_skill_prompts = pd.read_json('skill_prompts.json').reset_index(drop=True)
        train_skill_prompts = train_skill_prompts.rename(columns={'bloom_prompt': 'context'})
        train_skill_prompts = train_skill_prompts[['q_id', 'context']]

        train_df = train_df.merge(train_skill_prompts, on='q_id')[relevant_columns]
        train_df = train_df.rename(columns={"context": "instruction", "hypothesis": "output"})
        train_df['input'] = ''


        test_dialogs_ids = [
            '0abf5da615c94838af2221e89fd076b8',
            'deedc0a0f7e64c7fa1cd7b916b215df4',
            'ba3739ffb8fc4b2a82f4838810cd16d8',
            '1bf15c29e7fc46a98d1fef797b5d0956',
            '2f52f6482af04852ab785156eb1336b7',
            '3100792bb4184ed4b62e3d7208a96a01',
            '0fbfc8a3f48d4a1d99d6cab0ef04d2f5',
            '16d8de033dca4f6d9c1dfb01581358f9',
            '42eb2981c76546e6a2786ebd0ece5eb9',
            'f364d289f3af4f95a6ac542eaf1b88bd',
        ]

        test_dialogs_mask = train_df['dialog_id'].isin(test_dialogs_ids)

        test_df = train_df[test_dialogs_mask]
        train_df = train_df[~test_dialogs_mask]

        val_dialogs_ids = [
            '029886841f1142e5b75fec7d80c3b95a',
            '7565fbfda0fb495e99adddafa5269b31',
            'd044f44749f24993a7f084ce709b28e6',
            'd044f44749f24993a7f084ce709b28e6',
            'c4c35cadf93f48d19bcb2c74fe812fdd',
        ]
        val_dialogs_mask = train_df['dialog_id'].isin(val_dialogs_ids)

        val_df = train_df[val_dialogs_mask]
        train_df = train_df[~val_dialogs_mask]

        dataset_size = len(train_df) + len(val_df) + len(test_df)

        print(
            f"Train: {len(train_df) / dataset_size * 100:.1f}%\n"
            f"Val: {len(val_df) / dataset_size * 100:.1f}%\n"
            f"Test: {len(test_df) / dataset_size * 100:.1f}%"
        )

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    # Generate dataset from pandas dataframe
    train_data = Dataset.from_pandas(train_df).shuffle().map(generate_and_tokenize_prompt)
    val_data = Dataset.from_pandas(val_df).shuffle().map(generate_and_tokenize_prompt)
    test_data = Dataset.from_pandas(test_df).shuffle().map(generate_and_tokenize_prompt)

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParalldlism when more than 1 gpu is available
        print("[WARNING] More than 1 GPU, turn on DDP")
        model.is_parallelizable = True
        model.model_parallel = True

    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=5,
        optim="adamw_torch",
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=5,
        save_steps=5,
        output_dir=output_dir,
        save_total_limit=100,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=group_by_length,
        report_to="wandb" if use_wandb else None,
        run_name=wandb_run_name if use_wandb else None,
    )

    if optim == "sophia":
        # https://github.com/Liuhong99/Sophia/issues/17
        try:
            from Sophia.sophia import SophiaG
        except ImportError:
            raise ImportError(
                "Can't import sophia. You should clone sophia repo from https://github.com/Liuhong99/Sophia to project folder to use sophia")
        optimizer = SophiaG(model.parameters(), lr=training_args.learning_rate, betas=(0.9, 0.999), rho=0.03)

        trainer = transformers.Trainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=val_data,
            args=training_args,
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
            optimizers=(optimizer, None,)
        )
    else:
        trainer = transformers.Trainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=val_data,
            args=training_args,
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ))

    if early_stopping:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=20, early_stopping_threshold=0.01))

    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)
