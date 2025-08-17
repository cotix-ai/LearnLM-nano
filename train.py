# -*- coding: utf-8 -*-

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)

def main():
    # 定义模型和数据路径
    base_model_path = '/root/autodl-tmp/qwen/Qwen2.5-7B-Instruct'
    dataset_path = 'dataset/fine-tune.json'
    output_dir = "./output/LearnLM-nano"

    # --- 1. 加载和预处理数据集 ---
    print("--- Step 1: Loading and processing dataset ---")

    # 将JSON文件加载到Pandas DataFrame，然后转换为Hugging Face Dataset
    df = pd.read_json(dataset_path)
    ds = Dataset.from_pandas(df)

    # 打印前3条数据以供检查
    print("Sample data from the dataset:")
    print(ds[:3])

    # 加载分词器
    # 使用 trust_remote_code=True 是因为 Qwen2 模型结构需要执行模型仓库中的代码
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        use_fast=False,
        trust_remote_code=True
    )
    # Qwen2 tokenizer 的 pad_token 默认为 <|endoftext|>，是正常的
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token.")

    # # 处理数据集
    # 定义数据处理函数，将原始文本转换为模型输入格式
    def process_func(example):
        MAX_LENGTH = 384  # 设置最大长度
        
        # 构建符合 Qwen2-Instruct 聊天格式的 prompt
        # System Prompt 用于角色扮演
        system_prompt = "<|im_start|>system\n你是一个有用的人工智能教学助手 LearnLM。<|im_end|>\n"
        user_prompt = f"<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n"
        assistant_prompt = "<|im_start|>assistant\n"
        
        # 指令部分
        instruction = tokenizer(
            f"{system_prompt}{user_prompt}{assistant_prompt}",
            add_special_tokens=False  # 不在开头添加 special tokens
        )
        # 回答部分
        response = tokenizer(
            f"{example['output']}",
            add_special_tokens=False
        )

        # 组合 input_ids, attention_mask 和 labels
        input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.eos_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
        # labels 中，指令部分设置为-100，这样模型在计算损失时会忽略它们，只关注回答部分
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.eos_token_id]

        # 截断过长的数据
        if len(input_ids) > MAX_LENGTH:
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    # 对整个数据集应用处理函数
    tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)
    print("\nDataset processed. Example of tokenized data:")
    print("Decoded input_ids[0]:", tokenizer.decode(tokenized_ds[0]['input_ids']))
    print("Decoded labels[1] (filtered):", tokenizer.decode(list(filter(lambda x: x != -100, tokenized_ds[1]["labels"]))))

    # --- 2. 加载模型和 LoRA 配置 ---
    print("\n--- Step 2: Loading model and configuring LoRA ---")
    
    # 创建模型
    # 加载预训练模型，使用 bfloat16 以节省显存，并自动分配到可用设备
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    # 开启梯度检查点时，需要执行此方法
    model.enable_input_require_grads()
    print("\nModel loaded. Dtype:", model.dtype)

    # LoRA 配置
    # 配置 LoRA 参数
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,  # 设置为训练模式
        r=8,                   # LoRA 矩阵的秩
        lora_alpha=32,         # LoRA 缩放因子
        lora_dropout=0.1       # Dropout 比例
    )

    # 将 LoRA 配置应用到模型
    model = get_peft_model(model, lora_config)
    
    # 打印模型中可训练参数的比例
    print("\nLoRA configured. Trainable parameters:")
    model.print_trainable_parameters()
    
    # --- 3. 配置训练参数并开始训练 ---
    print("\n--- Step 3: Configuring and starting training ---")
    
    # # 配置训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,  # 实际 batch_size = 4 * 4 = 16
        logging_steps=10,
        num_train_epochs=3,
        save_steps=100,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,  # 开启梯度检查点，节省显存
    )

    # 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    # 开始训练
    print("\nStarting training...")
    trainer.train()
    print("Training complete.")
    
    # --- 4. 加载微调后的模型并进行测试 ---
    print("\n--- Step 4: Loading fine-tuned model for inference ---")
    
    # 清理显存
    del model
    del trainer
    torch.cuda.empty_cache()
    
    # 合并加载模型
    lora_checkpoint_path = f"{output_dir}/checkpoint-600" # 您可以根据实际情况修改为你保存的最佳checkpoint
    
    # 重新加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).eval()
    
    # 加载并融合 LoRA 权重
    # eval() 模式下，PeftModel 会自动合并权重用于推理
    model = PeftModel.from_pretrained(base_model, model_id=lora_checkpoint_path)
    print(f"\nLoaded fine-tuned model from {lora_checkpoint_path}")

    # 准备测试 prompt
    prompt = "你是谁？"
    # 使用 apply_chat_template 构造符合模型预期的输入格式
    messages = [
        {"role": "system", "content": "你是 LearnLM，一个有用的人工智能教学助手。"},
        {"role": "user", "content": prompt}
    ]
    
    # 将文本编码为 token
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to(model.device)

    # 设置生成参数
    gen_kwargs = {"max_new_tokens": 256, "do_sample": True, "top_k": 1}

    # 执行生成
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        response_ids = outputs[0, inputs['input_ids'].shape[1]:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

    print("\n--- Inference Test ---")
    print(f"Prompt: {prompt}")
    print(f"Generated Response: {response_text}")


if __name__ == '__main__':
    main()
