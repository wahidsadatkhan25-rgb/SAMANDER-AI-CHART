
import os
import base64
from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI

app = Flask(__name__, static_folder=".")
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

@app.route("/")
def home():
    return send_from_directory(".", "index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "online",
        "project": "SAMANDER AI OTC",
        "provider": "Groq"
    })

@app.route("/analyze", methods=["POST"])
def analyze():

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return jsonify({
            "error": "Groq API key is not configured."
        }), 503

    if "image" not in request.files:
        return jsonify({
            "error": "Please upload a chart screenshot."
        }), 400

    image = request.files["image"]
    pair = request.form.get("pair", "Unknown OTC pair")

    allowed = {
        "image/jpeg",
        "image/png",
        "image/webp"
    }

    if image.mimetype not in allowed:
        return jsonify({
            "error": "Use JPG, PNG or WEBP."
        }), 400

    image_bytes = image.read()

    if not image_bytes:
        return jsonify({
            "error": "Uploaded image is empty."
        }), 400

    encoded = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    image_url = (
        f"data:{image.mimetype};base64,{encoded}"
    )

    prompt = f"""
You are SAMANDER AI, a cautious chart screenshot
analysis assistant.

Market: {pair}
Timeframe: 1 minute.

Examine only the visible chart information.

Analyze:
1. Recent candle direction and structure.
2. Visible candlestick patterns.
3. Possible support and resistance.
4. MACD only if visible.
5. Bullish or bearish evidence.

Return this format:

ESTIMATE: BUY, SELL, or WAIT
TREND:
CANDLE PATTERN:
SUPPORT/RESISTANCE:
MACD:
REASON:
UNCERTAINTY:

Rules:
- Choose WAIT if the screenshot is unclear.
- Never invent unseen information.
- Never guarantee the next candle.
- Never invent confidence percentages.
- This is practice analysis, not a trading instruction.
"""

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )

        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                }
            ],
            temperature=0.2,
            max_completion_tokens=700
        )

        result = response.choices[0].message.content

        if not result:
            raise ValueError("Empty AI response")

        return jsonify({
            "success": True,
            "pair": pair,
            "timeframe": "1 minute",
            "analysis": result
        })

    except Exception:
        app.logger.exception("Groq analysis failed")

        return jsonify({
            "error": "Groq analysis failed. Check Render logs."
        }), 500


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "error": "Image too large. Maximum size is 8 MB."
    }), 413


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5000))
        )
    
