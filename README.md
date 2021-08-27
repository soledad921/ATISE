README
===========================
This is the source code of our papers: 

ISWC2020 paper: ["Temporal Knowledge Graph Completion based on Time Series Gaussian Embedding"](https://arxiv.org/pdf/1911.07893.pdf)

COLING2020 paper: ["TeRo: A Time-aware Knowledge Graph Embedding via Temporal Rotation"](https://arxiv.org/pdf/2010.01029.pdf)
****
## Implementation Environment:
* Python 3.7, Pytorch 1.0, CUDA 8.0, Anaconda 4.8.3
* Python 3.7, Pytorch 1.4, CUDA 9.1, Anaconda 4.5.11


## Dataset:
* The link of the original dataset YAGO11k can be found from paper: [HyTE: Hyperplane-based Temporally aware Knowledge Graph Embedding](https://github.com/malllabiisc/HyTE)
* The links of the original datasets ICEWS14 and ICEWS05-15 can be found from paper: [Learning Sequence Encoders for Temporal Knowledge Graph Completion](https://github.com/nle-ml/mmkb)
* We uniform the formats of all these datasets.

## Usage:
* Install dependencies and put dataset folders here  
* model.py contains PyTorch(1.x) based implementation of our proposed models
* To reproduce the reported results of our models, use the following commands:

      python Main.py --model TERO --dataset icews14 --dim 500 --lr 0.1 --gamma 110 --loss logloss --eta 10 --timedisc 0 --cuda True --gran 1

      python Main.py --model TERO --dataset icews05-15 --dim 500 --lr 0.1 --gamma 120 --loss logloss --eta 10 --timedisc 0 --cuda True --gran 2

      python Main.py --model TERO --dataset yago --dim 500 --lr 0.1 --gamma 50 --loss marginloss --timedisc 2 --cuda True --gran 1 --thre 100

      python Main.py --model TERO --dataset wikidata --dim 500 --lr 0.3 --gamma 20 --loss logloss --timedisc 2 --cuda True --gran 1 --thre 300



      python Main.py --model ATISE --dataset icews14 --dim 500 --lr 0.00003 --gamma 120 --loss logloss --timedisc 0 --cuda True --gran 3 --cmin 0.003

      python Main.py --model ATISE --dataset icews05-15 --dim 500 --lr 0.00003 --gamma 100 --loss logloss --timedisc 0 --cuda True --gran 30 --cmin 0.003

      python Main.py --model ATISE --dataset yago --dim 500 --lr 0.00003 --gamma 1 --loss logloss --timedisc 1 --cuda True --gran 1 --cmin 0.005 --thre 300

      python Main.py --model ATISE --dataset wikidata --dim 500 --lr 0.00003 --gamma 1 --loss logloss --timedisc 1 --cuda True --gran 1 --cmin 0.005 --thre 300

* Parameters and Some of the important available options include:  

	    task: [LinkPrediction,TimePrediction]	(default:LinkPrediction)	
	    model:  [ATISE,TERO]   (default: ATISE)
	    dataset: [icews14,icews05-15,yago,wikidata] (default: icews14)
	    max_epoch: (shoud be >500) (default: 5000)
	    dim: 	number of dimension (default: 500)
	    batch: 	batchsize (default:512)
	    lr: 	learning rate (default:0.1)
	    gamma: 	margin for translational models (default:1)
	    eta:	ratio of negative samples over the positives (default: 10)
	    timedisc: the method used for handling facts involving time intervals: 0 means no time intervals; 1 means to discretize time intervals into time points; 2 means to use dual relation embeddings (default: 0)
	    cuda:   whether to use cuda devices (default: True)
	    loss: use which loss function for optimization: logloss means logistic loss function; marginloss means margin rank loss (default: logloss)
	    cmin: minimum threshold of covariance matrices of ATISE (default: 0.005)
	    gran: the time unit of icews datasets (default: 1)
	    thre: the mini threshold of time classes in yago and wikidata (default: 1)

* Results will be printed out and stored in the corresponding dataset folders.

## Citation
If you use the codes, please cite the following papers:
* ATiSE:

      @inproceedings{ATiSE,
	  title={Temporal knowledge graph completion based on time series gaussian embedding},
	  author={Xu, Chenjin and Nayyeri, Mojtaba and Alkhoury, Fouad and Yazdi, Hamed and Lehmann, Jens},
	  booktitle={International Semantic Web Conference},
	  pages={654--671},
	  year={2020},
	  organization={Springer}
      }
      
* TeRo:

      @inproceedings{TERO,
	    title = "{T}e{R}o: A Time-aware Knowledge Graph Embedding via Temporal Rotation",
	    author = "Xu, Chengjin  and
	      Nayyeri, Mojtaba  and
	      Alkhoury, Fouad  and
	      Shariat Yazdi, Hamed  and
	      Lehmann, Jens",
	    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
	    month = dec,
	    year = "2020",
	    address = "Barcelona, Spain (Online)",
	    publisher = "International Committee on Computational Linguistics",
	    url = "https://aclanthology.org/2020.coling-main.139",
	    doi = "10.18653/v1/2020.coling-main.139",
	    pages = "1583--1593"
      }

## License
ATISE is MIT licensed, as found in the LICENSE file.
