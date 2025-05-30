**Raw data** of the three datasets (SHS27k, SHS148k, and STRING) can be downloaded from the [Google Drive](https://drive.google.com/file/d/1hJVrQXddB9JK68z7jlIcLfd9AmTWwgJr/view?usp=sharing):

* `protein.STRING.sequences.dictionary.tsv`      Protein sequences of STRING
* `protein.actions.STRING.txt`     PPI network of STRING
* `STRING_AF2DB`     PDB files of protein structures predicted by AlphaFold2

Pre-process raw data to generate feature and adjacency matrices (also applicable to any new dataset):
```
python ./raw_data/data_process.py --dataset data_name
```
where `data_name` is one of the three datasets (SHS27k, SHS148k, and STRING).
