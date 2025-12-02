./#! /bin/bash

# install torch and mpmath, typing-extensions, sympy, nvidia-nvtx-cu12, nvidia-nvjitlink-cu12, nvidia-nccl-cu12, nvidia-curand-cu12, nvidia-cufft-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, networkx, MarkupSafe, fsspec, filelock, triton, nvidia-cusparse-cu12, nvidia-cudnn-cu12, jinja2, nvidia-cusolver-cu12
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install bitsandbytes==0.44.1
pip install sentencepiece==0.2.0

# install local transformers and urllib3, tqdm, safetensors, regex, pyyaml, packaging, numpy, idna, charset-normalizer, certifi, requests, huggingface-hub, tokenizers
cd transformers
pip install -e .
cd ..

# install local torchao
cd ao
pip install -e .
cd ..

# install local lm-evaluation-harness and word2number, sqlitedict, pytz, zstandard, xxhash, tzdata, threadpoolctl, tcolorpy, tabulate, six, scipy, pybind11, pyarrow, psutil, propcache, portalocker, pathvalidate, numexpr, multidict, more_itertools, lxml, joblib, fsspec, frozenlist, dill, colorama, click, chardet, attrs, aiohappyeyeballs, absl-py, yarl, tqdm-multiprocess, scikit-learn, sacrebleu, python-dateutil, nltk, multiprocess, mbstrdecoder, jsonlines, aiosignal, typepy, rouge-score, pandas, aiohttp, accelerate, peft, datasets, DataProperty, tabledata, evaluate, pytablewriter
cd lm-evaluation-harness
pip install -e .
cd ..

cd accelerate
pip install -e . --upgrade
cd ..

cd peft
pip install -e . --upgrade
cd ..
