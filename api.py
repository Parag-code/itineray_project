from flask import Flask, request, jsonify, Response
from main import build_itinerary, make_human_like

app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate_itinerary():
    try:
        data = request.get_json()
        query = data.get("query")

        if not query:
            return jsonify({"error": "Missing 'query' field"}), 400

        # Step 1: Build itinerary
        parsed, structured = build_itinerary(query)

        # Step 2: Make human-like text
        human_like = make_human_like(parsed, structured)

        # Step 3: Return a combined response
        # JSON part for API, but human text as PRETTY string
        return jsonify({
            "structured": structured,
            "human_like": human_like.split("\n")  # split into array for readability in Postman
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
