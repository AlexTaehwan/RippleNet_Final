## Data Preprocessing
1. data/amazon-book, unzip kg_final.txt.zip
2. data/last-fm, unzip kg_final.txt.zip
3. data/yelp2018, unzip kg_final.txt.zip

4. data/amazon-book, python preprocess.py --> this will build kg_final.txt and ratings_final.txt
5. data/last-fm, python preprocess.py --> this will build kg_final.txt and ratings_final.txt
6. data/yelp2018, python preprocess.py --> this will build kg_final.txt and ratings_final.txt

## Run the main function

amazon-book : /src, python main.py --amazon-book --n_epoch 20

last-fm : /src, python main.py --last-fm --n_epoch 20

yelp2018 : /src, python main.py --yelp2018 --n_epoch 20


### all the environment settings are saved in requirements.txt
