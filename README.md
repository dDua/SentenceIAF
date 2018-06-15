This repository is adapted from https://github.com/timbmg/Sentence-VAE

# Sentence Inverse Autoregressive Flow

PyTorch implementation of [_Generating Sentences from a Continuous Space_](https://arxiv.org/abs/1511.06349) by Bowman et al. 2015.
_Note: This implementation does not support LSTM's at the moment, but RNN's and GRU's._

## Example results

### Performance on the Penn Tree Bank dataset
The true ELBO was optimized for approximately 1 epoch (as can bee see in the graph above). Results are averaged over entire split.

| Test | RE   | KL    | ELBO   | Perplexity|
|:------------|:------:|:-----:|:-----:|:-----:|
| Sentence VAE [[Bowman et al., 2016]](http://www.aclweb.org/anthology/K16-1002) | 99 | 2 | 101 | 119 |
| Sentence VAE | 106.3 | 0.1 | 106.4 | 129.2 |
| Planar Flow | 104.2 | 3.9 | 108.1 | 117.4 |
| Radial Flow | 98.9 | 11.6 | 110.5 | 92.4 |
| Linear IAF | 98.9 | 11.6 | 110.5 | 92.4 |
| ResNet IAF | 70.9 | 32.3 | 103.2 | 102.3 |
**RE**: Reconstruction Error, **KL**: Kullback-Leibler divergence, **ELBO**: Evidence Lower BOund

### Sampled sentences with beam search
Sentenes have been obtained after sampling from ***z*** ~ *N*(0, I).  

| Model | Beam Width | Sampled Sentence |
|:------------|------:|:-----|
| Sentence VAE | 1 | my husband said that you must be . |
| | 3 | my husband had a lot of time . |
| | 5 | my husband told me that i am . |
| | 15 | my husband told me that lord rafe ? |
| Planar Flow | 1 | i placed my hand on my shin and a pair of jeans . |
| | 3 | i placed my hand on my clothes and i read it . |
| | 5 | i placed my hand on my clothes and i read it . |
| | 15 | i put my socks and shoes and socks and shoes and socks . |
| Radial Flow | 1 | anna gave me a nod and said , i know you were still on the way . |
| | 3 | ive got a couple of months, i said , you know what I do . |
| | 5 | ive got a couple of months, and i dont know how to do it . |
| | 15 | give me a couple of times , and i dont know how i know . |


### Interpolating sentences
Sentences produced by greedily decoding from points betweentwo sentence encodings with a planar VAE model.  

**after a couple of months , i wondered how many of them had come to me .**  
after a couple of months , i had to admit it was a good idea .  
after a few moments of the first time , we had a lot .  
in the middle of the cavern was a long pause .  
**in the middle of the tunnel was a dazzling colour .**  

---

**and many of the people had been there for a while .**  
and many of the people had been here for you .  
and what have you done , i ?  
and you have a little chat ?  
**you have to do it ?**  

## Experiments
Please download the Penn Tree Bank data first (download from [Tomas Mikolov's webpage](http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz)). The code expects to find at least `ptb.train.txt` and `ptb.valid.txt` in the specified data directory. The data can also be donwloaded with the `dowloaddata.sh` script.

### Preprocessing
Vocabuluary needs to be extracted from training data
```
python3 utils/preprocess.py -input data/ptb.train.txt -output data/phb.vocab.txt
```


### Training
Then training can be executed with the following command:
```
python3 train.py --config config.yaml
```

You can switch models by changing **map_type** in `config.yaml`  
- **planar**: Planar Flows  
- **radial**: Radial Flows  
- **linear**: Linear IAF  
- **resnet**: ResNet IAF


### Evaluation
After training, you can evaluate the model with test data:
```
python3 evaluate.py --config config.yaml
```


### Inference
For obtaining samples and interpolating between senteces,
```
python3 inference.py -c $CHECKPOINT -n $NUM_SAMPLES
```

Note that samples cannot be generated in this manner for models inverse autoregressive flows,
since these modelsd epend on the deterministich variable ***h*** emitted by the encoder

