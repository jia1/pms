# pms

### Instructions for developers

If you already have a virtual environment, omit the `python3 -m venv .` command.

```bash
git clone https://github.com/jia1/pms.git
cd pms
python3 -m venv .
. bin/activate
pip3 install -r requirements.txt
```

Additional commands to run jupyter notebook in virtual environment:

```bash
ipython kernel install --user --name=pms
jupyter notebook
```
