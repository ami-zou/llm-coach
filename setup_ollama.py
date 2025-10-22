# setup_ollama.py - Setup script for Ollama with Chinese models
import requests
import subprocess
import time

def check_ollama_installed():
    """Check if Ollama is installed."""
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Ollama已安装: {result.stdout.strip()}")
            return True
        else:
            print("❌ Ollama未正确安装")
            return False
    except FileNotFoundError:
        print("❌ Ollama未安装。请访问 https://ollama.ai 下载安装")
        return False

def start_ollama_server():
    """Start Ollama server if not running."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print("✅ Ollama服务已在运行")
            return True
    except:
        pass
    
    print("🚀 启动Ollama服务...")
    try:
        # Start Ollama server in background
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)  # Wait for server to start
        
        # Check if it's running
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama服务启动成功")
            return True
        else:
            print("❌ Ollama服务启动失败")
            return False
    except Exception as e:
        print(f"❌ 启动Ollama服务时出错: {e}")
        return False

def get_available_models():
    """Get list of available models."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        return []
    except:
        return []

def pull_chinese_model():
    """Pull a good Chinese model."""
    chinese_models = [
        "qwen2.5:7b",  # Good Chinese model
        "deepseek-coder:6.7b",  # Good for coding and general tasks
        "llama3.2:3b",  # Smaller but capable
        "gemma2:9b",  # Google's model
        "deepseek-llm:7b" # Newest deepseek llm model
    ]
    
    print("📥 推荐的中文模型:")
    for i, model in enumerate(chinese_models, 1):
        print(f"  {i}. {model}")
    
    print("\n💡 建议使用 'qwen2.5:7b' 或 'deepseek-coder:6.7b'")
    print("   运行以下命令下载模型:")
    print("   ollama pull qwen2.5:7b")
    print("   或")
    print("   ollama pull deepseek-coder:6.7b")

def main():
    """Main setup function."""
    print("🔧 Ollama RAG系统设置")
    print("="*50)
    
    # Check if Ollama is installed
    if not check_ollama_installed():
        print("\n📥 请先安装Ollama:")
        print("   macOS: brew install ollama")
        print("   或访问: https://ollama.ai")
        return
    
    # Start Ollama server
    if not start_ollama_server():
        return
    
    # Get available models
    models = get_available_models()
    if models:
        print(f"\n📋 当前可用模型: {models}")
    else:
        print("\n📋 暂无模型，需要下载")
    
    # Suggest Chinese models
    pull_chinese_model()
    
    print("\n✅ 设置完成！")
    print("🚀 现在可以运行: python run_ollama_rag.py")

if __name__ == "__main__":
    main()
