# anchovy
The Anchovy Clickbait Detector by https://www.linkedin.com/in/alessandromacagno/ written in Python 3

Usage
1. Download training data: https://www.clickbait-challenge.org/#data and copy to /data and eval.py
2. git clone https://github.com/tira-io/anchovy
3. cd anchovy
4. pip install all dependencies
5. python3 train.py. You can edit flags in utils.py
6. python3 test.py -i test_data -o output
7. python3 eval.py test_data/truth.jsonl ./output/results.jsonl output.prototext 

If you don't have locally your test data and you want to evaluate on http://www.tira.io/.
1. ask for a Tira.io account to martin.potthast@uni-leipzig.de 
2. ssh <username>@<host> -p <port>
3. git clone, cd etc.
4. scp -P <port> -r data/runs/* <username>@<host>:anchovy/data/runs
5. Go to tira.io page and submit first program python3 anchovy/test.py -i $inputDataset -o $outputDir. Run it
6. Wait for approval and see your results
