# Development server, for debugging only
from dotenv import load_dotenv
from apps.main import app

if __name__ == "__main__":
    load_dotenv()
    app.run(debug=True)