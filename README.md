# DeepLncPro: an interpretable convolutional neural network model for identifying long non-coding RNA promoters

**Our [article](https://academic.oup.com/bib/article/23/6/bbac447/6754194) is now published in *Briefings in Bioinformatics*.**

This is an implementation of the published paper (DOI: [10.1093/bib/bbac447](https://doi.org/10.1093/bib/bbac447)). In this work, we propose a convolutional neural network based model, called DeepLncPro, to identify lncRNA promoters in human and mouse. Comparative results demonstrated that DeepLncPro was superior to both state-of-the-art machine learning methods and existing models for identifying lncRNA promoters.

![DeepLncPro overview](js/DeepLncPro_overview.png)

## Installation

- <span  style="color: #5bdaed; font-weight: bold">[Python](https://www.python.org/)>=3.8</span>
- [Pytorch](https://pytorch.org/)>=1.8.2
- [NumPy](https://numpy.org/)>=1.21.5
```
git clone https://github.com/zhangtian-yang/DeepLncPro.git
cd DeepLncPro
pip install -r requirements.txt
```

## Example

```
python DeepLncPro.py -i example.txt -o1 output_prediction.html -o2 output_motifcheck.html -s h -ts 0.5
```

## Optional arguments

```
-h, --help HELP
     Show help message and exit.
-i, --input INPUT
     The input file is a complete Fasta format sequence. default=input.txt
-o1, --output1 OUTPUT1
     Results of predicting lncRNA promoters. default=output_prediction.html
-o2, --output2 OUTPUT2
     Results of predicting motifs. default=output_motifcheck.html 
-s, --species SPECIES
     Choose the species of Human/Mouse. default=h
-ts, --threshold THRESHOLD
     Threshold for classification based on predicted scores. default=0.5
```

## Acknowledgements

DeepLncPro uses and/or references the following separate libraries and packages (ordered alphabetically):

- [deepRAM](https://github.com/MedChaabane/deepRAM)
- [NumPy](https://numpy.org/)
- [Pytorch](https://pytorch.org/)
- [Tomtom](https://meme-suite.org/meme/tools/tomtom)
  
Thanks for all their contributors and maintainers!
