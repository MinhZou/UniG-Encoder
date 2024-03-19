run example:

```
python train_optuna.py --method PlainUnigencoder --dname citeseer --epochs 500 --runs 10 --cuda 0 --data_dir ./processed_data/cocitation/citeseer  --raw_data_dir ./raw_data/cocitation/citeseer
python train_optuna.py --method PlainUnigencoder --dname cora --epochs 500 --runs 10 --cuda 0 --data_dir ./processed_data/cocitation/cora  --raw_data_dir ./raw_data/cocitation/cora
python train_optuna.py --method PlainUnigencoder --dname pubmed --epochs 500 --runs 10 --cuda 0 --data_dir ./processed_data/cocitation/pubmed  --raw_data_dir ./raw_data/cocitation/pubmed
python train_optuna.py --method PlainUnigencoder --dname coauthor_cora --epochs 500 --runs 10 --cuda 0 --data_dir ./processed_data/coauthorship/cora  --raw_data_dir ./raw_data/coauthorship/cora
python train_optuna.py --method PlainUnigencoder --dname coauthor_dblp --epochs 500 --runs 10 --cuda 0 --data_dir ./processed_data/coauthorship/dblp  --raw_data_dir ./raw_data/coauthorship/dblp
python train_optuna.py --method PlainUnigencoder --dname house-committees --epochs 500 --runs 10 --cuda 0 --data_dir ./processed_data/house-committees  --raw_data_dir ./raw_data/house-committees
python train_optuna.py --method PlainUnigencoder --dname senate-committees --epochs 500 --runs 10 --cuda 0 --data_dir ./processed_data/senate-committees --raw_data_dir ./raw_data/senate-committees
python train_optuna.py --method PlainUnigencoder --dname ModelNet40 --epochs 500 --runs 10 --cuda 0 --data_dir ./processed_data/ModelNet40 --raw_data_dir ./raw_data/ModelNet40
python train_optuna.py --method PlainUnigencoder --dname NTU2012 --epochs 500 --runs 10 --cuda 0 --data_dir ./processed_data/NTU2012 --raw_data_dir ./raw_data/NTU2012
python train_optuna.py --method PlainUnigencoder --dname 20newsW100 --epochs 500 --runs 10 --cuda 0 --data_dir ./processed_data/20newsW100 --raw_data_dir ./raw_data/20newsW100
python train_optuna.py --method PlainUnigencoder --dname zoo --epochs 500 --runs 10 --cuda 0 --data_dir ./processed_data/zoo --raw_data_dir ./raw_data/zoo
python train_optuna.py --method PlainUnigencoder --dname yelp --epochs 500 --runs 10 --cuda 0 --data_dir ./processed_data/yelp --raw_data_dir ./raw_data/yelp
```

