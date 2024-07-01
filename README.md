# DeepLncPro: an interpretable convolutional neural network model for identifying long non-coding RNA promoters

**Our [article](https://academic.oup.com/bib/article/23/6/bbac447/6754194) is now published in *Briefings in Bioinformatics*.**

DeepLncPro is able to identify the promoters of long non-coding RNA in human and mouse.
This is an implementation of the published paper "DeepLncPro: an interpretable convolutional neural network model for identifying long non-coding RNA promoters."

## Installation:

- <span  style="color: #5bdaed; font-weight: bold">[Python](https://www.python.org/)>=3.8</span>
- [Pytorch](https://pytorch.org/)>=1.8.2
- [NumPy](https://numpy.org/)>=1.21.5
```
git clone https://github.com/zhangtian-yang/DeepLncPro.git
cd DeepLncPro
pip install -r requirements.txt
``` 
## Optional arguments:

```
-h, --help HELP
     Show help message and exit.
-i, --input INPUT
     The input file is a complete Fasta format sequence. default=input.txt
-o1, --output1 OUTPUT1
     Results of predicting lncRNA promoters are saved under results folder. default=output_prediction.html
-o2, --output2 OUTPUT2
     Results of predicting motifs are saved under results folder. default=output_motifcheck.html 
-s, --species SPECIES
     Choose the species of Human/Mouse. default=h
-ts, --threshold THRESHOLD
     Threshold for classification based on predicted scores. default=0.5
```
## Example:

```
python DeepLncPro.py -i example.txt -o1 output_prediction.html -o2 output_motifcheck.html -s h -ts 0.5
```

## Acknowledgements

DeepLncPro uses and/or references the following separate libraries and packages (ordered alphabetically):

- [deepRAM](https://github.com/MedChaabane/deepRAM)
- [NumPy](https://numpy.org/)
- [Pytorch](https://pytorch.org/)
- [Tomtom](https://meme-suite.org/meme/tools/tomtom)
  
Thanks for all their contributors and maintainers!
