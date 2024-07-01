# DeepLncPro: an interpretable convolutional neural network model for identifying long non-coding RNA promoters

**Our [article](https://academic.oup.com/bib/article/23/6/bbac447/6754194) is now published in *Briefings in Bioinformatics*.**

DeepLncPro is able to identify the promoters of long non-coding RNA in human and mouse.
This is an implementation of the published paper "DeepLncPro: an interpretable convolutional neural network model for identifying long non-coding RNA promoters."

## Installation:

- <span  style="color: #5bdaed; font-weight: bold">python>=3.8</span>
- pytorch>=1.8.2
- numpy>=1.21.5
``` 
pip install torch numpy
``` 
``` 
pip install -r requirement.txt
``` 
## Optional arguments:

```
  -h, --help            Show this help message and exit.
  --addresses ADDRESSES
                        Tianyang.Zhang819@outlook.com
  -i INPUTFILE, --inputFile INPUTFILE
                        -i input.txt (The input file is a complete Fasta
                        format sequence.)
  -o1 OUTPUTFILE1, --outputFile1 OUTPUTFILE1
                        -o1 output_prediction.html (Results of predicting 
                        lncRNA promoters are saved under results folder.)
  -o2 OUTPUTFILE2, --outputFile2 OUTPUTFILE2
                        -o2 output_motifcheck.html (Results of predicting 
                        motifs are saved under results folder.)
  -s SPECIES, --species SPECIES
                        -s Human/Mouse (Choose one from two species to
                        use.)
  -ts THRESHOLD, --threshold THRESHOLD  
                        -ts 0.5(Prediction result threshold)
```
## Example:

```
python DeepLncPro.py -i example.txt -o1 output_prediction.html -o2 output_motifcheck.html -s h -ts 0.5
```

## Acknowledgements

DeepLncPro uses and/or references the following separate libraries and packages (ordered alphabetically):

- [Pytorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [deepRAM](https://github.com/MedChaabane/deepRAM)
- [Tomtom](https://meme-suite.org/meme/tools/tomtom)
  
Thanks for all their contributors and maintainers!

***
Version number：V1.2 <br>
Updated date：2022-08-30 <br>
Email: Tianyang.Zhang819@outlook.com
***
