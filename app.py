import os
import json
from flask import Flask, request, jsonify, render_template_string
from datetime import datetime

# Third-party SDK for Google's Gemini
try:
    import google.generativeai as genai
except Exception:
    genai = None

# ---------------------------
# Configuration
# ---------------------------
APP_NAME = "Genie"
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
DEFAULT_MODEL = os.environ.get("GENIE_MODEL", "gpt-4o-mini")  # change as needed

if genai is not None and GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception:
        # If configure fails, we'll attempt to re-configure per request
        pass

# ---------------------------
# Flask app
# ---------------------------
app = Flask(__name__)

# minimal in-memory conversation store (for demo only). For production, use a DB.
CONVERSATIONS = {}

INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Genie — Chat</title>
  <style>
    body { font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto; background:#0f172a; color:#e6eef8; display:flex; align-items:center; justify-content:center; height:100vh; margin:0 }
    .card { width:92%; max-width:800px; background:#071233; padding:18px; border-radius:12px; box-shadow: 0 10px 30px rgba(2,6,23,.6)}
    .messages { height:60vh; overflow:auto; padding:10px; border-radius:8px; background:linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); margin-bottom:12px }
    .msg { margin:8px 0; }
    .msg.user { text-align:right }
    .bubble { display:inline-block; padding:10px 14px; border-radius:12px; max-width:80% }
    .bubble.user { background:#0ea5a4; color:#012; }
    .bubble.bot { background:#1e293b; color:#e6eef8; }
    .input-row { display:flex; gap:8px }
    input[type=text]{ flex:1; padding:10px 12px; border-radius:8px; border:1px solid rgba(255,255,255,0.06); background:transparent; color:inherit }
    button{ padding:10px 14px; border-radius:8px; border:none; background:#3b82f6; color:white; cursor:pointer }
    .meta { font-size:12px; color:#9fb0d6 }
  </style>
</head>
<body>
  <div class="card">
    <h2>Genie</h2>
    <div class="meta">Model: <span id="model">{{ model }}</span></div>
    <div id="messages" class="messages"></div>

    <div class="input-row">
      <input id="uinput" type="text" placeholder="Say something to Genie..." />
      <button id="send">Send</button>
    </div>
  </div>

<script>
const messagesEl = document.getElementById('messages')
const input = document.getElementById('uinput')
const sendBtn = document.getElementById('send')

function appendMessage(text, cls){
  const div = document.createElement('div')
  div.className = 'msg ' + (cls || '')
  const bubble = document.createElement('div')
  bubble.className = 'bubble ' + (cls === 'user' ? 'user' : 'bot')
  bubble.textContent = text
  div.appendChild(bubble)
  messagesEl.appendChild(div)
  messagesEl.scrollTop = messagesEl.scrollHeight
}

sendBtn.addEventListener('click', async ()=>{
  const text = input.value.trim()
  if(!text) return
  appendMessage(text, 'user')
  input.value = ''
  appendMessage('Genie is thinking...', 'bot')

  try{
    const res = await fetch('/api/chat', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({message:text})
    })
    const data = await res.json()
    // remove the 'thinking' placeholder (last bot message)
    const placeholders = Array.from(document.querySelectorAll('.msg.bot'))
    if(placeholders.length) placeholders[placeholders.length-1].remove()

    if(data.error){
      appendMessage('Error: ' + data.error, 'bot')
    }else{
      appendMessage(data.reply, 'bot')
    }
  }catch(e){
    appendMessage('Network error. ' + e.message, 'bot')
  }
})

input.addEventListener('keydown', (e)=>{ if(e.key === 'Enter') sendBtn.click() })
</script>
</body>
</html>
"""

# ---------------------------
# Helper: call Gemini / Google Generative AI
# ---------------------------

def call_gemini(prompt, conversation_id=None, model_name=DEFAULT_MODEL):
    """
    Sends prompt to Gemini via google.generativeai SDK if available.

    Returns: (reply_text, raw_response)
    """
    # Try SDK first
    if genai is not None:
        try:
            # Some versions of the SDK use `chat.create`, others `responses.create` or `generate`.
            # We'll try common call patterns and fall back gracefully.

            # 1) chat.create pattern
            if hasattr(genai, 'chat') and hasattr(genai.chat, 'create'):
                messages = [{"author": "user", "content": prompt}]
                if conversation_id:
                    resp = genai.chat.create(model=model_name, messages=messages, conversation=conversation_id)
                else:
                    resp = genai.chat.create(model=model_name, messages=messages)
                # Try to extract text
                reply = None
                if hasattr(resp, 'last') and hasattr(resp.last, 'content'):
                    reply = resp.last.content
                elif isinstance(resp, dict):
                    # tolerant parsing
                    reply = resp.get('content') or resp.get('output') or json.dumps(resp)
                else:
                    reply = str(resp)
                return reply, resp

            # 2) responses.create / generate
            if hasattr(genai, 'responses') and hasattr(genai.responses, 'create'):
                resp = genai.responses.create(model=model_name, input=prompt)
                # extract textual output safely
                text = None
                if hasattr(resp, 'output'):
                    # resp.output could be a list of message parts
                    try:
                        # Try common property shapes
                        if isinstance(resp.output, list):
                            parts = [getattr(p, 'content', None) or p for p in resp.output]
                            text = '\n'.join([str(p) for p in parts if p])
                        else:
                            text = str(resp.output)
                    except Exception:
                        text = str(resp)
                else:
                    text = str(resp)
                return text, resp

            # 3) generate fallback
            if hasattr(genai, 'generate'):
                resp = genai.generate(model=model_name, input=prompt)
                # try to extract
                if hasattr(resp, 'output'):
                    out = resp.output
                    if isinstance(out, str):
                        return out, resp
                    else:
                        return json.dumps(out), resp
                return str(resp), resp

        except Exception as e:
            return None, {"error": str(e)}

    # If SDK is not available or failed, fail gracefully
    return None, {"error": "Generative AI SDK unavailable or API key not set."}

# ---------------------------
# Routes
# ---------------------------

@app.route('/')
def index():
    return render_template_string(INDEX_HTML, model=DEFAULT_MODEL)


@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.get_json(force=True)
    message = (data.get('message') or '').strip()
    if not message:
        return jsonify({"error": "no message provided"}), 400

    # Create a simple conversation id using a timestamp for demo.
    conversation_id = data.get('conversation_id') or f"conv-{datetime.utcnow().isoformat()}"

    # Call Gemini
    reply, raw = call_gemini(message, conversation_id=conversation_id)
    if reply is None:
        # If SDK failed, try to give an informative error
        err = raw.get('error') if isinstance(raw, dict) else 'unknown error'
        return jsonify({"error": f"Gemini request failed: {err}"}), 500

    # Save to in-memory store for demo
    CONVERSATIONS.setdefault(conversation_id, []).append({"role": "user", "text": message})
    CONVERSATIONS[conversation_id].append({"role": "assistant", "text": reply})

    return jsonify({"reply": reply})


# ---------------------------
# Health check for Render
# ---------------------------
@app.route('/health')
def health():
    return jsonify({"status": "ok", "app": APP_NAME})


# ---------------------------
# CLI entrypoint
# ---------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

