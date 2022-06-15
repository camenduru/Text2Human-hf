#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess

import gradio as gr

if os.getenv('SYSTEM') == 'spaces':
    subprocess.run('pip uninstall -y mmcv-full'.split())
    subprocess.run('pip install mmcv-full==1.5.2'.split())
    with open('patch') as f:
        subprocess.run('patch -p1'.split(), cwd='Text2Human', stdin=f)

from model import Model

DESCRIPTION = '''# Text2Human

This is an unofficial demo for <a href="https://github.com/yumingj/Text2Human">https://github.com/yumingj/Text2Human</a>.
You can modify sample steps and seeds. By varying seeds, you can sample different human images under the same pose, shape description, and texture description. The larger the sample steps, the better quality of the generated images. (The default value of sample steps is 256 in the original repo.)

Label image generation step can be skipped. However, in that case, the input label image must be 512x256 in size and must contain only the specified colors.
'''
FOOTER = '<img id="visitor-badge" alt="visitor badge" src="https://visitor-badge.glitch.me/badge?page_id=hysts.text2human" />'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--theme', type=str)
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    return parser.parse_args()


def set_example_image(example: list) -> dict:
    return gr.Image.update(value=example[0])


def set_example_text(example: list) -> dict:
    return gr.Textbox.update(value=example[0])


def main():
    args = parse_args()
    model = Model(args.device)

    with gr.Blocks(theme=args.theme, css='style.css') as demo:
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    input_image = gr.Image(label='Input Pose Image',
                                           type='pil',
                                           elem_id='input-image')
                    pose_data = gr.Variable()
                with gr.Row():
                    paths = sorted(pathlib.Path('pose_images').glob('*.png'))
                    example_images = gr.Dataset(components=[input_image],
                                                samples=[[path.as_posix()]
                                                         for path in paths])

                with gr.Row():
                    shape_text = gr.Textbox(
                        label='Shape Description',
                        placeholder=
                        '''<gender>, <sleeve length>, <length of lower clothing>, <outer clothing type>, <other accessories1>, ...
Note: The outer clothing type and accessories can be omitted.''')
                with gr.Row():
                    shape_example_texts = gr.Dataset(
                        components=[shape_text],
                        samples=[['man, sleeveless T-shirt, long pants'],
                                 ['woman, short-sleeve T-shirt, short jeans']])
                with gr.Row():
                    generate_label_button = gr.Button('Generate Label Image')

            with gr.Column():
                with gr.Row():
                    label_image = gr.Image(label='Label Image',
                                           type='numpy',
                                           elem_id='label-image')

                with gr.Row():
                    texture_text = gr.Textbox(
                        label='Texture Description',
                        placeholder=
                        '''<upper clothing texture>, <lower clothing texture>, <outer clothing texture>
Note: Currently, only 5 types of textures are supported, i.e., pure color, stripe/spline, plaid/lattice, floral, denim.'''
                    )
                with gr.Row():
                    texture_example_texts = gr.Dataset(
                        components=[texture_text],
                        samples=[['pure color, denim'], ['floral, stripe']])
                with gr.Row():
                    sample_steps = gr.Slider(10,
                                             300,
                                             value=10,
                                             step=10,
                                             label='Sample Steps')
                with gr.Row():
                    seed = gr.Slider(0, 1000000, value=0, step=1, label='Seed')
                with gr.Row():
                    generate_human_button = gr.Button('Generate Human')

            with gr.Column():
                with gr.Row():
                    result = gr.Image(label='Result',
                                      type='numpy',
                                      elem_id='result-image')

        gr.Markdown(FOOTER)

        input_image.change(fn=model.process_pose_image,
                           inputs=input_image,
                           outputs=pose_data)
        generate_label_button.click(fn=model.generate_label_image,
                                    inputs=[
                                        pose_data,
                                        shape_text,
                                    ],
                                    outputs=label_image)
        generate_human_button.click(fn=model.generate_human,
                                    inputs=[
                                        label_image,
                                        texture_text,
                                        sample_steps,
                                        seed,
                                    ],
                                    outputs=result)
        example_images.click(fn=set_example_image,
                             inputs=example_images,
                             outputs=example_images.components)
        shape_example_texts.click(fn=set_example_text,
                                  inputs=shape_example_texts,
                                  outputs=shape_example_texts.components)
        texture_example_texts.click(fn=set_example_text,
                                    inputs=texture_example_texts,
                                    outputs=texture_example_texts.components)

    demo.launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
