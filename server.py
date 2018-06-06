import sys
sys.path.append('./chainer-pix2pix')
import datetime
import numpy as np, pdb, os, cv2, argparse
from PIL import Image

from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver

import chainer
from chainer import serializers

# chainer-pix2pix
from model import Generator
from utils import data_process, output2img

parser = argparse.ArgumentParser()
# parser.add_argument('--model_file', required=True)
parser.add_argument('--port', default=8080, type=int)
parser.add_argument('--gpu', type=int, default=0, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--pix2pix_model_path', default='./chainer-pix2pix/result/G.npz', help='Path for pix2pix model')
args = parser.parse_args()

# Set up GAN G
G = Generator(64, 3)
serializers.load_npz(args.pix2pix_model_path, G)

if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
    G.to_gpu()                               # Copy the model to the GPU

img_shape = (256, 256, 3)

def translate(img):
    start = datetime.datetime.now()
    print("translation start")
    img_a = np.asarray(img, dtype=np.float32)
    img_a = np.transpose(img_a, (2, 0, 1))
    A = data_process([img_a], device=args.gpu)
    # with chainer.using_config('train', False): # TODO: activate.
    with chainer.using_config('enable_backprop', False):
        translated_a = np.squeeze(output2img(G(A)))
    print("translation end at ", datetime.datetime.now() - start)
    return translated_a

class Handler(BaseHTTPRequestHandler):
    def set_cors(self):
        self.send_header("access-control-allow-origin", "*")
        allow_headers = self.headers.get("access-control-request-headers", "*")
        self.send_header("access-control-allow-headers", allow_headers)
        self.send_header("access-control-allow-methods", "POST, OPTIONS")

    def do_OPTIONS(self):
        self.send_response(200)
        self.set_cors();

    def do_POST(self):

        print("post")
        data_string = self.rfile.read(int(self.headers['Content-Length']))
        bgr_img = cv2.imdecode(np.fromstring(data_string, dtype=np.uint8), 1)
        rgb_img = bgr_img[:, :, ::-1]
        translated_rgb = translate(rgb_img)
        translated_bgr = translated_rgb[:, :, ::-1]
        self.send_response(200)
        self.send_header('Content-type', 'image/png')
        self.set_cors();
        self.end_headers()
        self.wfile.write(cv2.imencode('.png', translated_bgr)[1].tostring())
        Image.fromarray(translated_rgb).save("translated.png")
        return

def run(port):
    server_address = ('', port)
    httpd = HTTPServer(server_address, Handler)
    httpd.serve_forever()
    print('httpd running...')
    sys.stdout.flush()

run(args.port)
