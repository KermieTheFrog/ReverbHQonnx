# @title Inference
import gradio as gr
from gradio import components
from pathlib import Path
import os
import json
import shutil
import time

with open('/content/drive/MyDrive/MDX_Colab/model_data.json', 'r') as f:
  model_data = json.load(f)

tracks_path = '/content/drive/MyDrive/MDX_Colab/tracks/'
separated_path = 'separated/'

def ReverbHQ(input_path,input_path2,  Denoise, Normalize, Chunks, Shifts):
    start_time = time.time()
    track = Path(str(input_path.name) if input_path else input_path2)
    track = track.stem + track.suffix
    track = tracks_path+track
    try:
      shutil.copyfile(str(input_path.name) if input_path else input_path2, track)
    except Exception as e: print(e)
    

    denoise = Denoise
    normalise = Normalize

    amplitude_compensation = model_data["(de)Reverb HQ By FoxJoy"]["compensate"]
    dim_f = model_data["(de)Reverb HQ By FoxJoy"]["mdx_dim_f_set"]
    dim_t = model_data["(de)Reverb HQ By FoxJoy"]["mdx_dim_t_set"]
    n_fft = model_data["(de)Reverb HQ By FoxJoy"]["mdx_n_fft_scale_set"]

    mixing_algorithm = 'min_mag'
    chunks = Chunks
    shifts = Shifts

    ##validate values
    normalise = '--normalise' if normalise else ''
    denoise = '--denoise' if denoise else ''
    margin = 44100
    os.chdir(path="/content/drive/MyDrive/MDX_Colab/")
    os.system(f"python main.py --n_fft {n_fft} --dim_f {dim_f} --dim_t {dim_t} --margin {margin} -i \"{track}\" --mixing {mixing_algorithm} --onnx \"{'onnx/(de)Reverb HQ By FoxJoy'}\" --model off  --shifts {round(shifts)} --stems v --invert v --chunks {chunks} --compensate {amplitude_compensation} {normalise} {denoise}")
    os.remove(track)
    return (f"--------------------------------------------------\nSuccessfully completed music demixing.\nTime Taken: {time.time()-start_time:.1f} Seconds\nResults: {(str(input_path.name) if input_path else input_path2)}\n--------------------------------------------------")

def MDX23(input_path, input_path2, ChunkSize, vocalsOnly):
    start_time = time.time()
    folder_path = (str(input_path.name) if input_path else input_path2)
    output_folder = "/content/drive/MyDrive/output"

    filename =  Path(folder_path).stem
    Path(output_folder,filename).mkdir(parents=True, exist_ok=True)
    os.chdir(path = "/content/MVSEP-MDX23-Colab")
    os.system(f'python inference.py --large_gpu --chunk_size {ChunkSize} --input_audio "{folder_path}" --vocals_only "{vocalsOnly}" --output_folder "{output_folder}"/"{filename}"')
    return (f"--------------------------------------------------\nSuccessfully completed audio seperation.\nTime Taken: {time.time()-start_time:.1f} Seconds\nResults: {output_folder}/{filename}\n--------------------------------------------------")

with gr.Blocks() as demo:
    with gr.Tab("MDX23"):
        with gr.Row():
            input_path = components.File()
            input_path2 = components.Textbox(label="Enter File Path", info="Only use if your file is already uploaded to google drive")

        chunk_size = components.Slider(minimum=-100000, maximum=500000, label="Chunk Size",info = "Use a lower Chunk Size if you are having memory issues.", value = 500000)
        vocalsOnly = components.Checkbox(label="Vocals Only", info = "Check this to only Split the Vocals and Instrumental. This may speed up processing.", show_label=True)
        Convert = components.Button()
        Output = components.Textbox(label="Output")
        Convert.click(fn = MDX23, inputs= [input_path, input_path2,chunk_size, vocalsOnly], outputs=[Output] )

    with gr.Tab("ReverbHQ by FoxJoy"):
        with gr.Row():
            input_path = components.File()
            input_path2 = components.Textbox(label="Enter File Path", info="Only use if your file is already uploaded to google drive")
        Denoise = components.Checkbox(label="Denoise", value=True)
        Normalize = components.Checkbox(label="Normalize")
        Chunks = components.Slider(label="Chunks", minimum=1, maximum=55, value=55)

        Shifts = components.Slider(label="Shifts", minimum=0, maximum=10, value=10)
        Convert = components.Button()
        Output = components.Textbox(label="Output")
        Convert.click(fn = ReverbHQ, inputs= [input_path, input_path2, Denoise, Normalize, Chunks, Shifts], outputs=[Output] )

demo.queue().launch(share = True, debug= True)
