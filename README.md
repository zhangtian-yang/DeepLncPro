DeepLncPro
====
Description:
------------
DeepLncPro is able to identify the promoters of long non-coding RNA in Human and Mouse.
More details are in DOI: 10.1093/bib/bbac447.
Installation:
-------------
- <span  style="color: #5bdaed; font-weight: bold">python3.8</span>
- pytorch==1.8.2+cpu
- numpy==1.21.5
``` 
pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio===0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cpu
pip3 install numpy==1.21.5
``` 
``` 
pip install -r yours/requirement.txt
``` 
Optional arguments:
-------------------
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
Example:
--------
```
python DeepLncPro.py -i example.txt -o1 output_prediction.html -o2 output_motifcheck.html -s h -ts 0.5
```
***
Version number：V0.2.0 <br>
Updated date：2022-08-30 <br>
Email: Tianyang.Zhang819@outlook.com
***
