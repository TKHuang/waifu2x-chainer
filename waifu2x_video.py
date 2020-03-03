import os
import time
import six
import sys
import argparse

import numpy as np
import chainer
import ffmpeg

from PIL import Image
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Process, Queue

from lib import iproc
from lib import reconstruct
from lib import srcnn
from lib import utils


def denoise_image(cfg, src, model):
    dst, alpha = split_alpha(src, model)
    # six.print_(
    #     'Level {} denoising...'.format(cfg.noise_level), end=' ', flush=True)
    if cfg.tta:
        dst = reconstruct.image_tta(dst, model, cfg.tta_level, cfg.block_size,
                                    cfg.batch_size)
    else:
        dst = reconstruct.image(dst, model, cfg.block_size, cfg.batch_size)
    if model.inner_scale != 1:
        dst = dst.resize((src.size[0], src.size[1]), Image.LANCZOS)
    # six.print_('OK')
    if alpha is not None:
        dst.putalpha(alpha)
    return dst


def upscale_image(cfg, src, scale_model, alpha_model=None):
    dst, alpha = split_alpha(src, scale_model)
    for i in range(int(np.ceil(np.log2(cfg.scale_ratio)))):
        # six.print_('2.0x scaling...', end=' ', flush=True)
        model = scale_model if i == 0 or alpha_model is None else alpha_model
        if model.inner_scale == 1:
            dst = iproc.nn_scaling(dst, 2)  # Nearest neighbor 2x scaling
            alpha = iproc.nn_scaling(alpha, 2)  # Nearest neighbor 2x scaling
        if cfg.tta:
            dst = reconstruct.image_tta(dst, model, cfg.tta_level,
                                        cfg.block_size, cfg.batch_size)
        else:
            dst = reconstruct.image(dst, model, cfg.block_size, cfg.batch_size)
        if alpha_model is None:
            alpha = reconstruct.image(alpha, scale_model, cfg.block_size,
                                      cfg.batch_size)
        else:
            alpha = reconstruct.image(alpha, alpha_model, cfg.block_size,
                                      cfg.batch_size)
        # six.print_('OK')
    dst_w = int(np.round(src.size[0] * cfg.scale_ratio))
    dst_h = int(np.round(src.size[1] * cfg.scale_ratio))
    if dst_w != dst.size[0] or dst_h != dst.size[1]:
        # six.print_('Resizing...', end=' ', flush=True)
        dst = dst.resize((dst_w, dst_h), Image.LANCZOS)
        # six.print_('OK')
    if alpha is not None:
        if alpha.size[0] != dst_w or alpha.size[1] != dst_h:
            alpha = alpha.resize((dst_w, dst_h), Image.LANCZOS)
        dst.putalpha(alpha)
    return dst


def split_alpha(src, model):
    alpha = None
    if src.mode in ('L', 'RGB', 'P'):
        if isinstance(src.info.get('transparency'), bytes):
            src = src.convert('RGBA')
    rgb = src.convert('RGB')
    if src.mode in ('LA', 'RGBA'):
        six.print_('Splitting alpha channel...', end=' ', flush=True)
        alpha = src.split()[-1]
        rgb = iproc.alpha_make_border(rgb, alpha, model)
        six.print_('OK')
    return rgb, alpha


def load_models(cfg):
    ch = 3 if cfg.color == 'rgb' else 1
    if cfg.model_dir is None:
        model_dir = 'models/{}'.format(cfg.arch.lower())
    else:
        model_dir = cfg.model_dir

    models = {}
    flag = False
    if cfg.method == 'noise_scale':
        model_name = 'anime_style_noise{}_scale_{}.npz'.format(
            cfg.noise_level, cfg.color)
        model_path = os.path.join(model_dir, model_name)
        if os.path.exists(model_path):
            models['noise_scale'] = srcnn.archs[cfg.arch](ch)
            chainer.serializers.load_npz(model_path, models['noise_scale'])
            alpha_model_name = 'anime_style_scale_{}.npz'.format(cfg.color)
            alpha_model_path = os.path.join(model_dir, alpha_model_name)
            models['alpha'] = srcnn.archs[cfg.arch](ch)
            chainer.serializers.load_npz(alpha_model_path, models['alpha'])
        else:
            flag = True
    if cfg.method == 'scale' or flag:
        model_name = 'anime_style_scale_{}.npz'.format(cfg.color)
        model_path = os.path.join(model_dir, model_name)
        models['scale'] = srcnn.archs[cfg.arch](ch)
        chainer.serializers.load_npz(model_path, models['scale'])
    if cfg.method == 'noise' or flag:
        model_name = 'anime_style_noise{}_{}.npz'.format(
            cfg.noise_level, cfg.color)
        model_path = os.path.join(model_dir, model_name)
        if not os.path.exists(model_path):
            model_name = 'anime_style_noise{}_scale_{}.npz'.format(
                cfg.noise_level, cfg.color)
            model_path = os.path.join(model_dir, model_name)
        models['noise'] = srcnn.archs[cfg.arch](ch)
        chainer.serializers.load_npz(model_path, models['noise'])

    if cfg.gpu >= 0:
        chainer.backends.cuda.check_cuda_available()
        chainer.backends.cuda.get_device(cfg.gpu).use()
        for _, model in models.items():
            model.to_gpu()
    return models


def extract_frame(queue, video, width, height, num_frames):
    process = (
        ffmpeg.input(video).output(
            'pipe:', format='rawvideo', pix_fmt='rgb24',
            vframes=num_frames).run_async(pipe_stdout=True))
    frame_size = width * height * 3
    for _ in range(num_frames):
        frame = process.stdout.read(frame_size)
        img = Image.fromarray(
            np.frombuffer(frame, dtype=np.uint8).reshape([height, width, 3]))
        while (queue.full()):
            time.sleep(3)
        queue.put(img)
    return 0


def main():
    p = argparse.ArgumentParser(description='Chainer implementation of waifu2x')
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--input', '-i', required=True)
    p.add_argument('--output', '-o', default=None)
    p.add_argument('--quality', '-q', type=int, default=None)
    p.add_argument('--model_dir', '-d', default=None)
    p.add_argument('--scale_ratio', '-s', type=float, default=2.0)
    p.add_argument('--tta', '-t', action='store_true')
    p.add_argument('--batch_size', '-b', type=int, default=16)
    p.add_argument('--block_size', '-l', type=int, default=128)
    p.add_argument('--extension', '-e', default='png', choices=['png', 'webp'])
    p.add_argument(
        '--arch',
        '-a',
        default='VGG7',
        choices=[
            'VGG7', '0', 'UpConv7', '1', 'ResNet10', '2', 'UpResNet10', '3'
        ])
    p.add_argument(
        '--method',
        '-m',
        default='scale',
        choices=['noise', 'scale', 'noise_scale'])
    p.add_argument(
        '--noise_level', '-n', type=int, default=1, choices=[0, 1, 2, 3])
    p.add_argument('--color', '-c', default='rgb', choices=['y', 'rgb'])
    p.add_argument('--tta_level', '-T', type=int, default=8, choices=[2, 4, 8])
    g = p.add_mutually_exclusive_group()
    g.add_argument('--width', '-W', type=int, default=0)
    g.add_argument('--height', '-H', type=int, default=0)
    g.add_argument('--shorter_side', '-S', type=int, default=0)
    g.add_argument('--longer_side', '-L', type=int, default=0)

    args = p.parse_args()
    if args.arch in srcnn.table:
        args.arch = srcnn.table[args.arch]

    models = load_models(args)

    try:
        probe = ffmpeg.probe(args.input)
    except ffmpeg.Error as e:
        print(e.stderr, file=sys.stderr)
        sys.exit(1)

    video_stream = next((stream for stream in probe['streams']
                         if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    num_frames = int(video_stream['nb_frames'])
    # parse fps info  ex: '25/1' to int 25.
    fps = video_stream['avg_frame_rate']
    fps = fps.split('/')
    fps = float(fps[0]) / float(fps[1])

    try:
        # extract frame in another process.
        q_in = Queue(maxsize=60)
        p_in = Process(
            target=extract_frame,
            args=(q_in, args.input, width, height, num_frames))
        p_in.start()

        # start output process.
        outext = Path(args.input).suffix
        out_wh = (int(args.scale_ratio * width), int(args.scale_ratio * height))
        if args.output is None:
            args.output = f'{args.input}_{args.arch}_{args.method}{args.scale_ratio}{args.noise_level}{outext}'
        process_out = (
            ffmpeg.input(
                'pipe:',
                format='rawvideo',
                pix_fmt='rgb24',
                framerate=fps,
                s='{}x{}'.format(out_wh[0], out_wh[1]))
            # .filter('fps', fps=args.fps, round='up')
            .output(
                args.output,
                pix_fmt='yuv420p',
                s='{}x{}'.format(
                    out_wh[0],
                    out_wh[1],
                )).overwrite_output().run_async(pipe_stdin=True))

        if args.width != 0:
            args.scale_ratio = args.width / width
        elif args.height != 0:
            args.scale_ratio = args.height / height
        elif args.shorter_side != 0:
            if width < height:
                args.scale_ratio = args.shorter_side / width
            else:
                args.scale_ratio = args.shorter_side / height
        elif args.longer_side != 0:
            if width > height:
                args.scale_ratio = args.longer_side / width
            else:
                args.scale_ratio = args.longer_side / height

        for _ in tqdm(range(num_frames)):
            while (q_in.empty()):
                time.sleep(3)
            src = q_in.get()

            dst = src.copy()
            if 'noise_scale' in models:
                dst = upscale_image(args, dst, models['noise_scale'],
                                    models['alpha'])
            else:
                if 'noise' in models:
                    dst = denoise_image(args, dst, models['noise'])
                if 'scale' in models:
                    dst = upscale_image(args, dst, models['scale'])

            dst.convert(src.mode)
            process_out.stdin.write(dst.tobytes())
    except KeyboardInterrupt:
        p_in.terminate()
        p_in.join()

    p_in.join()


if __name__ == '__main__':
    main()