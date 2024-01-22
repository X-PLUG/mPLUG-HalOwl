#python -m llava.serve.controller --host 0.0.0.0 --port 10000
#python -m llava.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload --port 5615
python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 5615 --worker http://localhost:5615 --model-path output/llava-7b-v1.3-sft_itc
