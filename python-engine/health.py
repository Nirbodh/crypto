from http.server import HTTPServer, BaseHTTPRequestHandler
import os

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Quant AI Engine Running")

PORT = int(os.environ.get("PORT", 8000))

server = HTTPServer(("0.0.0.0", PORT), HealthHandler)
print(f"Health server running on {PORT}")
server.serve_forever()