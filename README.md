## Create environment

- Python 3.10.13

```bash
conda create -n your_env_name python=3.10.13
```

- torch 2.1.1 + cu118

```bash
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
```

- Requirements: requirements.txt

```bash
pip install -r requirements.txt
```

- install `mamba`

```bash
pip install -e mamba
```

```bash
pip install -e caduceus
```
