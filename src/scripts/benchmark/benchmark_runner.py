import subprocess

def run_benchmarkers():
    print('running benchmark')
    with open('./queries/test_queries.txt', 'r') as file:
        for line in file:
            query = line.strip()
            subprocess.run(['python', 'query_benchmarker.py', query])

if __name__ == '__main__':
    run_benchmarkers()