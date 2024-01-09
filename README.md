# edit-gpt

For MacOS, you can create environment by running 
```bash
conda create -n ldm python=3.9 && activate ldm && conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 -c pytorch
```

Then run
```bash
pip install -r requirements.txt
```
(set global proxy if you are using)

If the error has occurred:

"OSError: Could not find library geos_c or load any of its variants ['/Library/Frameworks/GEOS.framework/Versions/Current/GEOS', '/opt/local/lib/libgeos_c.dylib', '/usr/local/lib/libgeos_c.dylib', '/opt/homebrew/lib/libgeos_c.dylib']"

you can install geos as follow:
```bash
sudo apt-get install geos # linux
brew install geos # macos
```

