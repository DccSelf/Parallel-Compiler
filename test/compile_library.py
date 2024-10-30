import subprocess
import argparse

def compile_and_archive(a_option=False):
    if not a_option:
        compile_command = ["clang", "-fPIC", "-c", "sylib.c", "-o", "sylib.o"]
        subprocess.run(compile_command, check=True)
        archive_command = ["ar", "rcs", "libsysy.a", "sylib.o"]
        subprocess.run(archive_command, check=True)
    else:
        # arm
        compile_command = ["arm-linux-gnueabihf-gcc", "-c", "sylib.c", "-o", "sylib.o"]
        subprocess.run(compile_command, check=True)
        archive_command = ["arm-linux-gnueabihf-ar", "rcs", "libsysy.a", "sylib.o"]
        subprocess.run(archive_command, check=True)
        runlib_command = ["arm-linux-gnueabihf-ranlib", "libsysy.a"]
        subprocess.run(runlib_command, check=True)

    print("Runtime library has been generated successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile and archive the runtime library.")
    parser.add_argument("-a", action="store_true", help="Compile for ARM architecture")
    
    args = parser.parse_args()
    
    compile_and_archive(args.a)