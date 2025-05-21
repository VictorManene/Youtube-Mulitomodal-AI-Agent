#import libraries
import os
import re
import gradio as gr
import main
from main import (
    process_video,
    download_audio_only,
    create_agent_system,
)
from langchain_openai import ChatOpenAI

# Custom CSS 
custom_css = """
:root {
    --primary-color: #4f46e5;
    --secondary-color: #10b981;
    --accent-color: #f59e0b;
    --success-color: #22c55e;
    --error-color: #ef4444;
    --warning-color: #f59e0b;
    --info-color: #3b82f6;

    /* Background colors */
    --background-color: #f8fafc;
    --card-bg: #ffffff;
    --card-border: rgba(226, 232, 240, 0.8);
    --card-hover: #f1f5f9;

    /* Text colors */
    --text-primary: #1e293b;
    --text-secondary: #64748b;
    --text-tertiary: #94a3b8;

    /* Other */
    --border-radius: 12px;
    --shadow-sm: 0 1px 2px 0 rgba(0,0,0,0.05);
    --shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
    --shadow-md: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05);
    --shadow-lg: 0 20px 25px -5px rgba(0,0,0,0.1), 0 10px 10px -5px rgba(0,0,0,0.04);
    --transition: all 0.3s ease;
}

/* Dark Mode Support */
.dark-mode {
    --background-color: #0f172a;
    --card-bg: #1e293b;
    --card-border: rgba(30, 41, 59, 0.8);
    --card-hover: #334155;
    --text-primary: #f8fafc;
    --text-secondary: #cbd5e1;
    --text-tertiary: #94a3b8;
}

/* Content Container */
.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 24px;
}

/* Main Layout */
#main-content {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
}

/* Header & Navigation */
.navbar-container {
    position: sticky;
    top: 0;
    z-index: 100;
    backdrop-filter: blur(10px);
    background: rgba(255, 255, 255, 0.8);
    border-bottom: 1px solid var(--card-border);
    transition: var(--transition);
}
.dark-mode .navbar-container {
    background: rgba(15, 23, 42, 0.8);
}
.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 24px;
    max-width: 1200px;
    margin: 0 auto;
}
.logo {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--primary-color);
}
.nav-links {
    display: flex;
    gap: 24px;
}
.nav-link {
    color: var(--text-secondary);
    text-decoration: none;
    font-weight: 500;
    padding: 8px 0;
    position: relative;
    transition: var(--transition);
}
.nav-link:hover {
    color: var(--primary-color);
}
.nav-link::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    width: 0;
    height: 2px;
    background: var(--primary-color);
    transition: var(--transition);
}
.nav-link:hover::after {
    width: 100%;
}
.nav-buttons {
    display: flex;
    gap: 12px;
}

/* Hero Section */
.hero {
    position: relative;
    padding: 120px 0 160px;
    background-size: cover;
    background-position: center;
    color: white;
    overflow: hidden;
}
.hero-overlay {
    position: absolute;
    inset: 0;
    background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.4));
    z-index: 1;
}
.hero-content {
    position: relative;
    z-index: 2;
    max-width: 800px;
    margin: 0 auto;
    text-align: center;
    padding: 0 24px;
}
.hero-title {
    font-size: 3.5rem;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 24px;
    color: #f3f4f6;
}
.hero-subtitle {
    font-size: 1.25rem;
    margin-bottom: 40px;
    color: rgba(255,255,255,0.9);
    max-width: 600px;
    margin-left: auto;
    margin-right: auto;
}
.hero-cta {
    display: flex;
    gap: 16px;
    justify-content: center;
}

/* Button styles with hover effects */
.btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 12px 24px;
    font-weight: 600;
    border-radius: 8px;
    transition: var(--transition);
    gap: 8px;
    cursor: pointer;
    position: relative;
    overflow: hidden;
    text-decoration: none;
}
.btn::after {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 5px;
    height: 5px;
    background: rgba(255, 255, 255, 0.5);
    opacity: 0;
    border-radius: 50%;
    transform: translate(-50%, -50%) scale(1);
}
.btn:focus:not(:active)::after {
    animation: ripple 1s ease-out;
}
.btn:hover {
    transform: scale(1.03);
    opacity: 0.95;
}
@keyframes ripple {
    0% { transform: scale(0); opacity: 1; }
    20% { transform: scale(25); opacity: 0.8; }
    100% { transform: scale(50); opacity: 0; }
}
.btn-primary {
    background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
    color: white;
    border: none;
    box-shadow: 0 4px 10px rgba(79, 70, 229, 0.25);
}
.btn-primary:hover {
    box-shadow: 0 6px 15px rgba(79, 70, 229, 0.35);
    transform: translateY(-2px) scale(1.05);
}
.btn-secondary {
    background: white;
    color: var(--primary-color);
    border: 1px solid var(--primary-color);
}
.btn-secondary:hover {
    background: rgba(79, 70, 229, 0.05);
    transform: translateY(-2px) scale(1.05);
}
.btn-text {
    background: transparent;
    color: var(--primary-color);
    padding: 8px 16px;
    font-weight: 500;
    text-decoration: none;
}
.btn-text:hover {
    background: rgba(79, 70, 229, 0.05);
    transform: scale(1.03);
}
.btn-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 40px;
    height: 40px;
    border-radius: 50%;
    padding: 0;
}
/* Footer Styles */
.footer {
    padding: 64px 0;
    background: var(--background-color);
    border-top: 1px solid var(--card-border);
}
.footer-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 24px;
}
.footer-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 32px;
    margin-bottom: 48px;
}
.footer-column {
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.footer-title {
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 16px;
}
.footer-link {
    color: var(--text-secondary);
    text-decoration: none;
    transition: var(--transition);
}
.footer-link:hover {
    color: var(--primary-color);
}
.footer-bottom {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
    border-top: 1px solid var(--card-border);
    padding-top: 24px;
}
.footer-copy {
    color: var(--text-tertiary);
    font-size: 0.875rem;
}
.footer-social {
    display: flex;
    gap: 16px;
}
.social-icon {
    width: 32px;
    height: 32px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: var(--card-hover);
    border-radius: 50%;
    transition: var(--transition);
    color: var(--text-secondary);
}
.social-icon:hover {
    background: var(--primary-color);
    color: white;
}

/* Feature Buttons Hover Enhancements */
.feature-button:hover {
    background-color: var(--card-hover);
    transform: scale(1.05);
    box-shadow: var(--shadow);
    transition: var(--transition);
}
/* Hover Card Enhancement */
.hover-card {
    transition: var(--transition);
    cursor: pointer;
}
.hover-card:hover {
    transform: translateY(-10px) scale(1.05);
    box-shadow: var(--shadow-lg);
}
.modern-input-container {
    position: relative;
    margin-bottom: 24px;
    background: var(--card-bg);
    border-radius: var(--border-radius);
    box-shadow: var(--shadow-sm);
    transition: var(--transition);
    overflow: hidden;
}

.modern-input-container:focus-within {
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
}

.modern-input-container label {
    position: absolute;
    left: 16px;
    top: 18px;
    color: var(--text-secondary);
    font-weight: 500;
    pointer-events: none;
    transition: var(--transition);
    font-size: 14px;
}

.modern-input-container input:focus ~ label,
.modern-input-container input:not(:placeholder-shown) ~ label,
.modern-input-container textarea:focus ~ label,
.modern-input-container textarea:not(:placeholder-shown) ~ label {
    top: 8px;
    font-size: 12px;
    color: var(--primary-color);
}

.modern-input-container input,
.modern-input-container textarea {
    width: 100%;
    padding: 24px 16px 12px 16px;
    font-size: 16px;
    border: 1px solid var(--card-border);
    border-radius: var(--border-radius);
    background: var(--card-bg);
    transition: var(--transition);
}

.modern-input-container input:focus,
.modern-input-container textarea:focus {
    outline: none;
    border-color: var(--primary-color);
}

.modern-input-container .input-icon {
    position: absolute;
    right: 16px;
    top: 50%;
    transform: translateY(-50%);
    color: var(--text-tertiary);
    transition: var(--transition);
}

.modern-input-container input:focus ~ .input-icon,
.modern-input-container input:not(:placeholder-shown) ~ .input-icon {
    color: var(--primary-color);
}

/* Analysis Card Styling */
.analysis-card {
    background: var(--card-bg);
    border-radius: var(--border-radius);
    border: 1px solid var(--card-border);
    padding: 16px;
    margin-bottom: 24px;
    transition: var(--transition);
    box-shadow: var(--shadow-sm);
}

.analysis-card:hover {
    box-shadow: var(--shadow);
}

.analysis-card-header {
    display: flex;
    align-items: center;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--card-border);
}

.analysis-card-icon {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-right: 12px;
    font-size: 14px;
    background: rgba(79, 70, 229, 0.1);
    color: var(--primary-color);
}

.analysis-card-title {
    font-weight: 600;
    color: var(--text-primary);
    font-size: 14px;
}

.analysis-card-content {
    font-size: 14px;
    color: var(--text-secondary);
    max-height: 200px;
    overflow-y: auto;
    padding-right: 8px;
}

.analysis-card-content::-webkit-scrollbar {
    width: 4px;
}

.analysis-card-content::-webkit-scrollbar-thumb {
    background-color: var(--text-tertiary);
    border-radius: 4px;
}

/* Modern Chat Interface */
.modern-chat-container {
    border-radius: var(--border-radius);
    border: 1px solid var(--card-border);
    overflow: hidden;
    margin: 24px 0;
    box-shadow: var(--shadow-sm);
    background: var(--card-bg);
}

.modern-chat-header {
    padding: 16px;
    border-bottom: 1px solid var(--card-border);
    display: flex;
    align-items: center;
    background: rgba(79, 70, 229, 0.05);
}

.modern-chat-title {
    font-weight: 600;
    color: var(--text-primary);
    margin-left: 12px;
}

.modern-chat-body {
    max-height: 300px;
    overflow-y: auto;
    padding: 16px;
}

.modern-chat-input-container {
    padding: 12px;
    border-top: 1px solid var(--card-border);
    display: flex;
    align-items: center;
    gap: 12px;
    background: rgba(248, 250, 252, 0.5);
}

.modern-chat-input {
    flex: 1;
    padding: 12px 16px;
    border-radius: 24px;
    border: 1px solid var(--card-border);
    background: var(--card-bg);
    transition: var(--transition);
}

.modern-chat-input:focus {
    outline: none;
    border-color: var(--primary-color);
    box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.1);
}

.modern-chat-send {
    background: var(--primary-color);
    color: white;
    width: 40px;              
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    border: none;
    cursor: pointer;
    transition: var(--transition);
    font-size: 20px;           
    line-height: 1;  
}


.modern-chat-send:hover {
    transform: scale(1.05);
    box-shadow: 0 2px 8px rgba(79, 70, 229, 0.25);
}

/* Modern Combined Card */
.analyzer-card {
    background: var(--card-bg);
    border-radius: var(--border-radius);
    border: 1px solid var(--card-border);
    box-shadow: var(--shadow);
    overflow: hidden;
    margin-bottom: 32px;
}

.analyzer-card-header {
    padding: 20px;
    background: linear-gradient(90deg, rgba(79, 70, 229, 0.1), rgba(16, 185, 129, 0.1));
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.analyzer-card-title {
    font-weight: 700;
    color: var(--text-primary);
    font-size: 18px;
}

.analyzer-card-body {
    padding: 24px;
}

.modern-input-group {
    display: flex;
    align-items: center;
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: var(--border-radius);
    overflow: hidden;
    transition: var(--transition);
}

.modern-input-group:focus-within {
    border-color: var(--primary-color);
    box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.1);
}

.modern-input-group-icon {
    padding: 0 16px;
    color: var(--text-tertiary);
    display: flex;
    align-items: center;
}

.modern-input-group input {
    flex: 1;
    border: none;
    padding: 16px;
    font-size: 15px;
    background: transparent;
}

.modern-input-group input:focus {
    outline: none;
}

/* Audio Player Styling */
.modern-audio-player {
    background: var(--card-bg);
    border-radius: var(--border-radius);
    padding: 16px;
    border: 1px solid var(--card-border);
    box-shadow: var(--shadow-sm);
    margin-bottom: 24px;
}

.modern-audio-header {
    display: flex;
    align-items: center;
    margin-bottom: 12px;
}

.modern-audio-title {
    font-weight: 600;
    margin-left: 12px;
    color: var(--text-primary);
}

.modern-audio-controls {
    padding: 8px;
    background: rgba(79, 70, 229, 0.05);
    border-radius: 8px;
}

/* Remove the wave SVG */
.wave { display: none; }
"""

# Session state
session = {"authenticated": False, "last_url": None, "transcript": None, "chat_history": []}
env_agent = None

# Authentication
main.simulate_login()
main.is_logged_in = True
main.ensure_authenticated()
env_agent = create_agent_system()

# Validator
YOUTUBE_REGEX = r"^(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+$"
def is_valid_youtube_url(url):
    return bool(re.match(YOUTUBE_REGEX, url))

# Video analysis
def multimodal_analyze(url, question):
    clean = url.strip().strip('"').strip("'")
    if not is_valid_youtube_url(clean):
        return "⚠️ Invalid YouTube URL", "", "", None
    transcript, ocr_data, answer, summary, audio_path = process_video(clean, question, play_audio=True)
    session['last_url'], session['transcript'] = clean, transcript
    session['chat_history'] = [{"role": "assistant", "content": answer}]
    ocr_str = "\n".join(f"{t:.1f}s: {txt}" for t, txt in ocr_data) if ocr_data else "⚠️ No on-screen text detected"
    return transcript, ocr_str, summary, audio_path

# Initialize LLM
a_pi = os.getenv("YOUTUBE_API_KEY")
if not a_pi: raise ValueError("Set OPENAI_API_KEY env var")
llm = ChatOpenAI(model="llama3-8b-8192",temperature=0.0,openai_api_key=a_pi,openai_api_base="https://api.groq.com/openai/v1")

# Chat functions
def multimodal_chat(user_input):
    if not session.get('transcript'): return [],""
    session['chat_history'].append({"role":"user","content":user_input})
    messages=[{"role":"system","content":session['transcript']}] + session['chat_history']
    try: ans=llm.invoke(messages).content
    except: ans="⚠️ Sorry, I can't follow up right now."
    session['chat_history'].append({"role":"assistant","content":ans})
    return session['chat_history'],""

def agent_analyze(url, user_input):
    clean = url.strip()
    if not is_valid_youtube_url(clean):
        return [{"role": "assistant", "content": "⚠️ Invalid YouTube URL for agent."}], ""
    session['last_url'] = clean
    audio_path, vid = download_audio_only(clean)
    tool_input = f"AUDIO_PATH:{audio_path}|VID_ID:{vid}"
    if user_input.lower().startswith("summarize"):
        tool = next((t for t in env_agent.tools if t.name == "SummarizeContent"), None)
    else:
        tool = next((t for t in env_agent.tools if t.name == "ReadSlides"), None)
    if not tool:
        ans = "⚠️ Tool not available"
    else:
        result = tool.run(tool_input)
        ans = result.get('output') if isinstance(result, dict) and 'output' in result else str(result)
    return [{"role": "user", "content": user_input}, {"role": "assistant", "content": ans}], ""

# Build UI
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue="blue",secondary_hue="green")) as demo:
    # Navbar
    with gr.Row(elem_id="navbar"):
        gr.Markdown("<div class='logo'>VideoInsight AI</div>")
        with gr.Row(elem_id="nav-buttons"):
            signin_btn=gr.Button("Sign In",elem_id='signin_btn')
            signup_btn=gr.Button("Sign Up",elem_id='signup_btn')
            back_btn=gr.Button("Back",visible=False)
    # Enhanced Hero
    gr.HTML("""
    <!-- Enhanced Hero Section -->
    <div class='hero' style='background-image:url("https://images.unsplash.com/photo-1516321318423-f06f85e504b3");background-size:cover;background-position:center;color:white;padding:120px 24px;'>
      <div style='position:absolute;inset:0;background:linear-gradient(rgba(0,0,0,0.7),rgba(0,0,0,0.4));'></div>
      <div style='position:relative;z-index:2;max-width:800px;margin:0 auto;text-align:center;'>
        <h2 class='hero-title' style='font-size:3rem;margin-bottom:24px;'>Unlock insights from YouTube videos</h2>
        <p style='font-size:1.2rem;margin-bottom:40px;color:rgba(255,255,255,0.9);'>Extract meaningful data with our advanced multimodal AI and Agent.</p>
        <div class='cta-buttons'>
          <a href='#analyzer' class='btn btn-primary'>Get started</a>
          <a href='#features' class='btn btn-secondary'>Learn more</a>
        </div>
      </div>
    </div>
    """)


    # Enhanced Features
    gr.HTML("""
    <div id='features' class='section bg-white' style='padding:80px 24px;'>
      <div style='max-width:1200px;margin:0 auto;'>
        <h3 class='gradient-text' style='text-align:center;font-size:2rem;margin-bottom:20px;'>Complete YouTube video analysis</h3>
        <p style='text-align:center;max-width:600px;margin:0 auto 60px;color:var(--text-secondary);'>Our platform combines state-of-the-art AI for comprehensive analysis.</p>
        <div style='display:flex;gap:24px;justify-content:center;flex-wrap:wrap;'>
          <div class='card hover-card' style='flex:1;min-width:250px;border-radius:20px;padding:30px;'>
            <div style='background:rgba(59,130,246,0.1);width:60px;height:60px;border-radius:50%;display:flex;align-items:center;justify-content:center;margin-bottom:20px;'><span style='font-size:24px;'>🎙️</span></div>
            <h4 style='font-size:1.2rem;margin-bottom:16px;'>Audio Transcription</h4>
            <p style='color:var(--text-secondary);'>Convert speech to text with Whisper for near-human accuracy.</p>
          </div>
          <div class='card hover-card' style='flex:1;min-width:250px;border-radius:20px;padding:30px;'>
            <div style='background:rgba(16,185,129,0.1);width:60px;height:60px;border-radius:50%;display:flex;align-items:center;justify-content:center;margin-bottom:20px;'><span style='font-size:24px;'>👁️</span></div>
            <h4 style='font-size:1.2rem;margin-bottom:16px;'>Visual Analysis</h4>
            <p style='color:var(--text-secondary);'>Extract key frames and identify objects, scenes, and visual contexts.</p>
          </div>
          <div class='card hover-card' style='flex:1;min-width:250px;border-radius:20px;padding:30px;'>
            <div style='background:rgba(245,158,11,0.1);width:60px;height:60px;border-radius:50%;display:flex;align-items:center;justify-content:center;margin-bottom:20px;'><span style='font-size:24px;'>🧠'</span></div>
            <h4 style='font-size:1.2rem;margin-bottom:16px;'>AI-Powered Summaries</h4>
            <p style='color:var(--text-secondary);'>Generate comprehensive summaries combining audio and visual insights.</p>
          </div>
        </div>
      </div>
    </div>
    """
    )
    # Analyzer
    
    with gr.Column(visible=True, elem_id="analyzer") as multimodal_view:
        with gr.Column(elem_classes="analyzer-card"):
            gr.HTML("""
            <div class="analyzer-card-header">
                <div class="analyzer-card-title">Video Analysis</div>
            </div>
            """)

            with gr.Column(elem_classes="analyzer-card-body"):
                # Replaced raw HTML input + hidden Textbox with a single Gradio Textbox
                url_input = gr.Textbox(
                    label="YouTube URL",
                    placeholder="Enter a valid YouTube link",
                    elem_id="url-input",
                    interactive=True,
                    elem_classes="modern-input",
                )
                

            # Custom Question Input
            with gr.Column(elem_classes="analyzer-card-body"):
                question_input = gr.Textbox(
                    label="Custom Question",
                    placeholder="Ask anything about the video...",
                    elem_id="question-input",
                    interactive=True,
                    elem_classes="modern-input",
                )

            analyze_btn = gr.Button("Analyze Video", elem_id="analyze-btn", elem_classes="btn btn-primary")

        # Results: Transcript
        with gr.Column(elem_classes="analysis-card"):
            gr.HTML('<div class="analysis-card-header"><div class="analysis-card-icon">📝</div><div class="analysis-card-title">Transcript</div></div>')
            transcript_o = gr.Textbox(label="", interactive=False, lines=5, elem_id="transcript_o", elem_classes="analysis-card-content")
        # Results: OCR
        with gr.Column(elem_classes="analysis-card"):
            gr.HTML('<div class="analysis-card-header"><div class="analysis-card-icon">👁️</div><div class="analysis-card-title">OCR Results</div></div>')
            ocr_o = gr.Textbox(label="", interactive=False, lines=5, elem_id="ocr_o", elem_classes="analysis-card-content")
        # Results: Summary
        with gr.Column(elem_classes="analysis-card"):
            gr.HTML('<div class="analysis-card-header"><div class="analysis-card-icon">📊</div><div class="analysis-card-title">Summary</div></div>')
            summary_o = gr.Textbox(label="", interactive=False, lines=3, elem_id="summary_o", elem_classes="analysis-card-content")

        # Audio Player
        with gr.Column(elem_classes="analysis-card modern-audio-player"):
            gr.HTML('<div class="analysis-card-header"><div class="analysis-card-icon">🔊</div><div class="analysis-card-title">Audio</div></div>')
            tts_player = gr.Audio(label="", interactive=False, elem_id="tts_player")

        # Chat Section
        with gr.Column(elem_classes="analysis-card modern-chat-container"):
            gr.HTML('<div class="analysis-card-header"><div class="analysis-card-icon">💬</div><div class="analysis-card-title">Follow-up Questions</div></div>')
            chatbox = gr.Chatbot(elem_id="chatbox", type='messages', elem_classes="modern-chat-body")
            with gr.Row(elem_classes="modern-chat-input-container"):
                user_msg = gr.Textbox(
                    placeholder="Ask follow-up question...",
                    label="",
                    elem_classes="modern-chat-input",
                )
                send_btn = gr.Button("✈️", elem_classes="modern-chat-send")
    # Agent view
    with gr.Column(visible=False) as agent_view:
        agent_url = gr.Textbox(
            label="YouTube URL for Agent",
            placeholder="Paste a YouTube link here",
        )
        agent_input = gr.Textbox(
            label="Agent Question",
            placeholder="Ask the AI agent...",
        )
        agent_send = gr.Button("Send to Agent", elem_classes="btn btn-primary")
        agent_chatbot = gr.Chatbot(type='messages', elem_id="agent-chatbox")
        # Follow-up questions input
        agent_followup = gr.Textbox(
            label="Follow-up Question",
            placeholder="Ask a follow-up question...",
        )
        followup_btn = gr.Button("Send Follow-up", elem_classes="btn btn-primary")
    # Footer
    gr.HTML("""
    <footer class='footer'>
      <div class='footer-container'>
        <div class='footer-grid'>
          <div class='footer-column'>
            <h4 class='footer-title'>Product</h4>
            <a class='footer-link' href='#'>Features</a>
            <a class='footer-link' href='#'>Pricing</a>
            <a class='footer-link' href='#'>API</a>
          </div>
          <div class='footer-column'>
            <h4 class='footer-title'>Company</h4>
            <a class='footer-link' href='#'>About</a>
            <a class='footer-link' href='#'>Blog</a>
            <a class='footer-link' href='#'>Careers</a>
          </div>
          <div class='footer-column'>
            <h4 class='footer-title'>Support</h4>
            <a class='footer-link' href='#'>Help Center</n>
            <a class='footer-link' href='#'>Privacy</a>
            <a class='footer-link' href='#'>Terms</a>
          </div>
          <div class='footer-column'>
            <h4 class='footer-title'>Social</h4>
            <div class='footer-social'>
              <a href='#' class='social-icon'>🐦</a>
              <a href='#' class='social-icon'>📘</a>
              <a href='#' class='social-icon'>📸</a>
            </div>
          </div>
        </div>
        <div class='footer-bottom'>
          <div class='footer-copy'>© 2025 VideoInsight AI. All rights reserved.</div>
          <div>
            <a class='footer-link' href='#'>Privacy</a> · <a class='footer-link' href='#'>Terms</a>
          </div>
        </div>
      </div>
    </footer>
    """),
    # Interactions
    analyze_btn.click(fn=multimodal_analyze, inputs=[url_input, question_input], outputs=[transcript_o, ocr_o, summary_o, tts_player])
    send_btn.click(fn=multimodal_chat, inputs=[user_msg], outputs=[chatbox, user_msg])
    agent_send.click(fn=agent_analyze, inputs=[agent_url, agent_input], outputs=[agent_chatbot, agent_input])
    followup_btn.click(
        fn=lambda q: agent_analyze(session['last_url'], q),
        inputs=[agent_followup],
        outputs=[agent_chatbot, agent_followup]
    )
    # Navbar actions
    def show_agent():
        session['authenticated'] = True
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)
    signin_btn.click(show_agent, outputs=[multimodal_view, agent_view, back_btn])
    signup_btn.click(show_agent, outputs=[multimodal_view, agent_view, back_btn])
    back_btn.click(lambda: (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)), outputs=[multimodal_view, agent_view, back_btn])

    
js_snippet = """
<script>
const urlField    = document.getElementById("url-input");      // your HTML <input>
const gradioField = document.querySelector("[data-component-id='url-input'] textarea");
urlField.addEventListener("input", (e) => {
    if (gradioField) {
        gradioField.value = e.target.value;
        gradioField.dispatchEvent(new Event("input", { bubbles: true }));
    }
});
</script>
<script>
window.addEventListener('DOMContentLoaded', () => {
    setTimeout(() => {
        setupDarkMode();setupLoadingStates();setupParallax();
        setupScrollAnimations();setupTypingAnimation();setupGlassmorphism();
        setupMiniVisualizations();setupConfetti();
    },1000);
});
</script>
"""
gr.HTML(js_snippet)

demo.launch(share=True, inbrowser=True)