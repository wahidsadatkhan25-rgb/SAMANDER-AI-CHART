
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
    return jsonify({"status": "online", "project": "SAMANDER AI OTC"})

@app.route("/analyze", methods=["POST"])
def analyze():
    if not os.getenv("OPENAI_API_KEY"):
        return jsonify({"error": "AI API key is not configured."}), 503

    if "image" not in request.files:
        return jsonify({"error": "Please upload a chart screenshot."}), 400

    image = request.files["image"]
    pair = request.form.get("pair", "Unknown OTC pair")

    allowed = {
        "image/jpeg": "jpeg",
        "image/png": "png",
        "image/webp": "webp"
    }

    if image.mimetype not in allowed:
        return jsonify({"error": "Use a JPG, PNG or WEBP screenshot."}), 400

    image_bytes = image.read()

    if not image_bytes:
        return jsonify({"error": "The uploaded image is empty."}), 400

    encoded = base64.b64encode(image_bytes).decode("utf-8")
    mime = image.mimetype
    image_url = f"data:{mime};base64,{encoded}"

    prompt = f"""
You are SAMANDER AI, a cautious chart screenshot analysis assistant.

Market: {pair}
Timeframe: 1 minute
Task: Examine the uploaded Pocket Option OTC chart screenshot.

Analyze only what is actually visible:
1. Recent candle direction and structure.
2. Visible candlestick patterns.
3. Possible support and resistance zones.
4. MACD only if it is visible in the screenshot.
5. Whether bullish or bearish evidence is stronger.

Return your assessment in this format:

ESTIMATE: BUY, SELL, or WAIT
TREND: ...
CANDLE PATTERN: ...
SUPPORT/RESISTANCE: ...
MACD: ...
REASON: ...
UNCERTAINTY: ...

Rules:
- Choose WAIT if the screenshot is unclear or evidence conflicts.
- Never invent indicator values or unseen chart information.
- Do not claim certainty or guarantee the next candle.
- This is an uncertain visual estimate, not a trading instruction.
- Do not invent a confidence percentage.
"""

    try:
        client = OpenAI()

        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt
                        },
                        {
                            "type": "input_image",
                            "image_url": image_url,
                            "detail": "high"
                        }
                    ]
                }
            ],
            max_output_tokens=500
        )

        result = response.output_text

        return jsonify({
            "success": True,
            "pair": pair,
            "timeframe": "1 minute",
            "analysis": result
        })

    except Exception as error:
        app.logger.exception("AI analysis failed")
        return jsonify({
            "error": "Analysis failed. Check the API configuration and try again."
        }), 500


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({"error": "Image is too large. Maximum size is 8 MB."}), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
      
