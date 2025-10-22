# run_ollama_rag.py - RAG system using local Ollama
import requests
import json
from query_rag import retrieve_relevant_chunks

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "deepseek-llm:7b"  # You can change this to any model you have

def generate_answer_with_ollama(context_chunks, user_query):
    """Generate answer using local Ollama."""
    # Create context from retrieved chunks
    context = "\n\n".join([f"• {chunk}" for chunk in context_chunks])
    
    # Create prompt in Chinese
    prompt = f"""你是职业教练QQ，专门帮助解决人生问题。基于以下讲座笔记内容，请给出专业建议：

【相关笔记内容】
{context}

【用户问题】
{user_query}

【教练QQ的专业回答】请基于以上内容给出具体、实用的建议："""

    # Prepare request for Ollama
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 500
        }
    }
    
    try:
        # Send request to Ollama
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        return result.get("response", "抱歉，无法生成回答。").strip()
        
    except requests.exceptions.ConnectionError:
        return "❌ 无法连接到Ollama服务。请确保Ollama正在运行。"
    except requests.exceptions.Timeout:
        return "❌ 请求超时，请稍后重试。"
    except Exception as e:
        return f"❌ 错误: {str(e)}"

def check_ollama_connection():
    """Check if Ollama is running and get available models."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✅ Ollama连接成功！")
            print(f"📋 可用模型: {[model['name'] for model in models]}")
            return True
        else:
            print(f"❌ Ollama响应错误: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到Ollama。请确保Ollama正在运行：")
        print("   ollama serve")
        return False
    except Exception as e:
        print(f"❌ 连接Ollama时出错: {e}")
        return False

def main():
    """Main function to run the RAG system."""
    print("🚀 Starting Ollama RAG System...")
    
    # Check Ollama connection
    if not check_ollama_connection():
        return
    
    print(f"🤖 使用模型: {MODEL_NAME}")
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
            answer = generate_answer_with_ollama(chunks, query)
            
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
