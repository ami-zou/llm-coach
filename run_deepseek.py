import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from query_rag import retrieve_relevant_chunks

# Load model once
MODEL_NAME = "deepseek-ai/deepseek-llm-7b-chat"

def load_model():
    """Load the model and tokenizer."""
    print("🤖 Loading DeepSeek model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    # Load model with optimizations and offloading
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        offload_folder="./offload"
    )
    
    # Configure generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def generate_answer(context_chunks, user_query, model, tokenizer):
    """Generate answer using DeepSeek model."""
    # Create context from retrieved chunks
    context = "\n\n".join([f"• {chunk}" for chunk in context_chunks])
    
    # Create prompt in Chinese
    prompt = f"""你是职业教练QQ，专门帮助年轻女性解决情感和人生问题。基于以下讲座笔记内容，请给出专业建议：

【相关笔记内容】
{context}

【用户问题】
{user_query}

【教练QQ的专业回答】"""

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()

def main():
    """Main function to run the RAG system."""
    print("🚀 Starting DeepSeek RAG System...")
    
    # Load model
    model, tokenizer = load_model()
    
    print("\n" + "="*60)
    print("💬 欢迎使用QQ教练RAG系统！")
    print("💡 输入 'quit' 或 'exit' 退出")
    print("="*60)
    
    while True:
        try:
            # Get user query
            query = input("\n💬 请输入您的问题: ").strip()
            
            if query.lower() in ['quit', 'exit', '退出']:
                print("👋 再见！")
                break
                
            if not query:
                print("❌ 请输入有效问题")
                continue
            
            print(f"\n🔍 正在搜索相关内容...")
            
            # Retrieve relevant chunks
            chunks = retrieve_relevant_chunks(query, n_results=3)
            
            if not chunks:
                print("❌ 未找到相关内容")
                continue
            
            print(f"📚 找到 {len(chunks)} 个相关片段")
            print("🤖 正在生成回答...")
            
            # Generate answer
            answer = generate_answer(chunks, query, model, tokenizer)
            
            print(f"\n🤖 QQ教练的建议：")
            print("="*60)
            print(answer)
            print("="*60)
            
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 错误: {e}")
            continue

if __name__ == "__main__":
    main()

