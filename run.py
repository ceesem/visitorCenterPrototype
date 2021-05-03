from src.dash_app import generate_app
from annotationframeworkclient import FrameworkClient

client = FrameworkClient("minnie65_phase3_v1")
app = generate_app(client, app_type="dash")

if __name__ == "__main__":
    app.run_server()