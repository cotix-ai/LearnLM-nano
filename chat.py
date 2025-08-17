import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --- 1. 配置和模型加载 ---

st.set_page_config(
    page_title="LearnLM-nano Chat",
    page_icon="🤖",
    layout="wide"
)

st.sidebar.title("模型配置")
base_model_path = st.sidebar.text_input(
    "基础模型路径 (Base Model Path)",
    value="/root/autodl-tmp/qwen/Qwen2.5-7B-Instruct"
)
lora_checkpoint_path = st.sidebar.text_input(
    "LoRA 权重路径 (LoRA Checkpoint Path)",
    value="./output/LearnLM-nano/checkpoint-600" # 脚本中测试用的checkpoint
)

SYSTEM_PROMPT = "你是 LearnLM，一个有用的人工智能教学助手。"

@st.cache_resource
def load_model(base_path, lora_path):
    """
    加载基础模型和 LoRA 权重，并返回合并后的模型和分词器。
    """
    try:
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            base_path,
            trust_remote_code=True,
            use_fast=False
        )
        # Qwen2 tokenizer 的 pad_token 默认为 <|endoftext|>
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            base_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).eval()

        # 加载并融合 LoRA 权重
        # PeftModel 会自动处理权重合并，以便进行高效推理
        model = PeftModel.from_pretrained(base_model, model_id=lora_path)
        
        return model, tokenizer
    except Exception as e:
        st.error(f"模型加载失败，请检查路径是否正确: {e}")
        return None, None

# --- 2. UI 界面和聊天逻辑 ---

st.title("🤖 LearnLM-nano ChatBot")
# 加载模型和分词器
with st.spinner("正在加载 LearnLM-nano 模型，请稍候..."):
    model, tokenizer = load_model(base_model_path, lora_checkpoint_path)

if model and tokenizer:
    # 初始化聊天记录
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 显示历史聊天记录
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    # 在侧边栏添加一个清除对话的按钮
    if st.sidebar.button("清除对话历史"):
        st.session_state.messages = []
        st.rerun()

    # 接收用户输入
    if prompt := st.chat_input("你好，我是 LearnLM，有什么可以帮你的吗？"):
        # 将用户输入添加到聊天记录并显示
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 准备模型输入
        # 1. 创建包含系统提示和所有对话历史的列表
        #    注意：每次都将完整的历史记录传递给模型，以便它理解上下文
        chat_history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ] + st.session_state.messages
        
        # 2. 使用 apply_chat_template 格式化输入
        inputs = tokenizer.apply_chat_template(
            chat_history,
            add_generation_prompt=True, # 自动添加 assistant 角色的起始符
            tokenize=True,
            return_tensors="pt",
        ).to(model.device)

        # 3. 生成回复
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                # 设置生成参数
                gen_kwargs = {"max_new_tokens": 512, "do_sample": True, "top_k": 5, "top_p": 0.9}
                
                with torch.no_grad():
                    outputs = model.generate(input_ids=inputs, **gen_kwargs)
                    # 从模型输出中只解码新生成的部分
                    response_ids = outputs[0, inputs.shape[1]:]
                    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

            # 显示模型的回复
            st.markdown(response_text)
        
        # 将模型的回复也添加到聊天记录
        st.session_state.messages.append({"role": "assistant", "content": response_text})
else:
    st.warning("模型未能成功加载，请检查侧边栏中的路径配置。")
