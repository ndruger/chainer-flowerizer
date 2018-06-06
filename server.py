import sys
# sys.path.append('./chainer-neural-style')
import datetime
import numpy as np, pdb, os, cv2, argparse
from PIL import Image

from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver

import chainer
from chainer import serializers

from pix2pix.model import Generator as Pix2pixGenerator
from pix2pix.utils import data_process as pix2pix_data_process, output2img as pix2pix_output2img

from neural_style.model import ImageTransformer
# from neural_style.utils import im_deprocess_vgg as style_im_deprocess_vgg

parser = argparse.ArgumentParser()
parser.add_argument('--port', default=8080, type=int)
parser.add_argument('--gpu', type=int, default=0, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--pix2pix_model_path', default='./pix2pix/result/G.npz', help='Path for pix2pix generator model', required=True)
parser.add_argument('--fast_style_transfer_model_path', default='./neural_style/fast_style_result/transformer_iter.npz', help='Path for chainer-neural-style transformer model', required=True)
args = parser.parse_args()

pix2pixG = Pix2pixGenerator(64, 3)
serializers.load_npz(args.pix2pix_model_path, pix2pixG)

styleTrans = ImageTransformer(32, 3, 150, False)
serializers.load_npz(args.fast_style_transfer_model_path, styleTrans)

if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()
    pix2pixG.to_gpu()
    styleTrans.to_gpu()

img_shape = (256, 256, 3)

def pix2pix_translate(img):
    start = datetime.datetime.now()
    print("translation start")
    img_a = np.asarray(img, dtype=np.float32)
    img_a = np.transpose(img_a, (2, 0, 1))
    A = pix2pix_data_process([img_a], device=args.gpu)
    # with chainer.using_config('train', False): # TODO: activate.
    with chainer.using_config('enable_backprop', False):
        translated_a = np.squeeze(pix2pix_output2img(pix2pixG(A)))
    print("translation end at ", datetime.datetime.now() - start)
    return translated_a

def original_colors(original, stylized):
    y, _, _ = cv2.split(cv2.cvtColor(stylized, cv2.COLOR_BGR2YUV))
    _, u, v = cv2.split(cv2.cvtColor(original, cv2.COLOR_BGR2YUV))
    yuv_img = cv2.merge((y, u, v))
    bgr_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR)
    return bgr_img

def style_translate(img):
    start = datetime.datetime.now()
    print("style start")
    img_a = np.asarray(img, dtype=np.float32)
    img_a = np.transpose(img_a, (2, 0, 1))
    A = chainer.cuda.to_cpu(styleTrans(np.asarray([img_a])).data)
    translated_a = np.asarray(np.transpose(A[0], [1, 2, 0]) + np.array([103.939, 116.779, 123.68]), dtype=np.uint8)
    translated_a = original_colors(img, translated_a)
    print("style end at ", datetime.datetime.now() - start)
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
        print("post: ", self.path)
        if self.path == '/pix2pix':
            self.do_pix2pix()
        elif self.path == '/styled':
            self.do_styled()
        else:
            print("unexpected path")
        return

    def do_pix2pix(self):
        data_string = self.rfile.read(int(self.headers['Content-Length']))
        bgr_img = cv2.imdecode(np.fromstring(data_string, dtype=np.uint8), 1)
        rgb_img = bgr_img[:, :, ::-1]
        translated_rgb = pix2pix_translate(rgb_img)
        translated_bgr = translated_rgb[:, :, ::-1]
        self.send_response(200)
        self.send_header('Content-type', 'image/png')
        self.set_cors();
        self.end_headers()
        self.wfile.write(cv2.imencode('.png', translated_bgr)[1].tostring())
        Image.fromarray(translated_rgb).save("translated.png")

    def do_styled(self):
        data_string = self.rfile.read(int(self.headers['Content-Length']))
        bgr_img = cv2.imdecode(np.fromstring(data_string, dtype=np.uint8), 1)
        rgb_img = bgr_img[:, :, ::-1]
        translated_rgb = style_translate(rgb_img)
        translated_bgr = translated_rgb[:, :, ::-1]
        self.send_response(200)
        self.send_header('Content-type', 'image/png')
        self.set_cors();
        self.end_headers()
        self.wfile.write(cv2.imencode('.png', translated_bgr)[1].tostring())
        Image.fromarray(translated_rgb).save("styled.png")

def run(port):
    server_address = ('', port)
    httpd = HTTPServer(server_address, Handler)
    print('staring server...')
    httpd.serve_forever()
    print('httpd running...')
    sys.stdout.flush()

run(args.port)
