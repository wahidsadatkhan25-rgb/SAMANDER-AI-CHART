
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
You are SAMANDER AI, a chart screenshot prediction assistant.

Market: {pair}
Timeframe: 1 minute.

Analyze the visible chart carefully:
- Recent price movement
- Candle direction and structure
- Visible candlestick patterns
- Support and resistance
- Momentum and trend

Predict the direction of the NEXT 1-MINUTE CANDLE.

Choose the stronger directional possibility based on
the visible chart evidence.

Your entire response MUST be exactly one of these:

ESTIMATE: BUY

OR

ESTIMATE: SELL

Do not include explanations, extra words, percentages,
guarantees, or other text.

Never invent chart information.
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
            max_completion_tokens=50
        )

        result = response.choices[0].message.content

        if not result:
            raise ValueError("Empty AI response")

        result = result.strip().upper()

        if result not in ["ESTIMATE: BUY", "ESTIMATE: SELL"]:
            raise ValueError("AI returned an invalid prediction.")

        return jsonify({
            "success": True,
            "pair": pair,
            "timeframe": "1 minute",
            "analysis": result
        })

    except Exception:
        app.logger.exception("Groq analysis failed")

        return jsonify({
            "error": "Analysis failed. Check Render logs."
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
    
