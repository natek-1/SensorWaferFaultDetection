# Steps to run
Welcome to my small but somewhat useful sensorfault detection project


Brief: In electronics, a wafer (also called a slice or substrate) is a thin slice of semiconductor, such as a crystalline silicon (c-Si), used for the fabrication of integrated circuits and, in photovoltaics, to manufacture solar cells. The wafer serves as the substrate(serves as foundation for contruction of other components) for microelectronic devices built in and upon the wafer.

It undergoes many microfabrication processes, such as doping, ion implantation, etching, thin-film deposition of various materials, and photolithographic patterning. Finally, the individual microcircuits are separated by wafer dicing and packaged as an integrated circuit.

Dataset is taken from Kaggle and stored in mongodb


1. Setup an enviroment on local machine to run project
```
conda create -p env python=3.8 -y
```
2. Activate enviroment
```
conda activate ./env/
```
3. Install requirments as setup
```
pip install -r requirements.txt
```
4. Run application
```
python app.py
```

🔧 Built with
- Flask
- Python 3.8
- XGBOOST
- Scikit learn
- Numpy
- Matploblib
- Seaborn


