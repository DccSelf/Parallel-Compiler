import os, platform, subprocess, shutil, sys

def build():
    build_dir = "../build/"

    if not os.path.exists(build_dir):
        os.mkdir(build_dir)
    else:
        shutil.rmtree(build_dir)
        os.mkdir(build_dir)

    os.system("cd ../build && cmake .. && make -j8")
        
if __name__ == "__main__":
    build()