# anchovy
The Anchovy Clickbait Detector by https://www.linkedin.com/in/alessandromacagno/ written in Python 3

Usage
1. Download training data: https://www.clickbait-challenge.org/#data and copy to /data
2. git clone https://github.com/tira-io/anchovy
3. cd anchovy
4. pip install all dependencies
5. python3 train.py. You can edit flags in utils.py
6. python3 test.py -i test_data -o output
7. python3 eval.py test_data/truth.jsonl ./output/predictions.jsonl output.prototext 
