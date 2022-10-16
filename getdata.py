import tarfile
import wget 
import os

# download File
def load_from_url(urls,path):
    filename = wget.download(url = urls ,out = path)
    print(f"LOADED {filename}")
    return filename

# tarfile unzip
def tar_extract(output_directory, filename):
    file_path = os.path.join(output_directory, filename)
    tar = tarfile.open(file_path, 'r:gz')
    tar.extractall(output_directory)

def main():
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    output_directory = 'DATA'
    filename = load_from_url(urls = url,path = output_directory)
    tar_extract(output_directory, filename)


if __name__ == '__main__':
    main()
