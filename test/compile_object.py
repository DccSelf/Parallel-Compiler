import os, subprocess, shutil

def compile_object(a_option=False):
    record = {}
    
    testcase_case = "./testcase/"
    output_case = "./output_o/"
    compiler_path = "../build/sysypc"
    
    if not os.path.exists(compiler_path):
        print("请先编译sysypc")
        exit()
    
    if not os.path.exists(output_case):
        os.mkdir(output_case)
    
    # 输出.o文件
    for i in [
        "functional", "hidden_functional", 
        "performance", "final_performance",
        "TestSysy", "TestTensor"
    ]:
        testcase_dir = testcase_case + i + '/'
        output_dir = output_case + i + '/'
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)
        if os.path.exists(testcase_dir):
            files = os.listdir(testcase_dir)
            src_files = sorted([f for f in files if f.endswith(".sy")])
            compiled_files = set()
            for src in src_files:
                fname, ftype = src.split('.')
                base_name = fname[:-1]
                if base_name in compiled_files:
                    # 检查是否有对应的 .o 文件
                    if os.path.exists(output_dir + base_name + ".o"):
                        print(f"Copy {base_name + '.o'} -> {fname + '.o'}")
                        shutil.copy(output_dir + base_name + ".o", output_dir + fname + ".o")
                        continue
                    elif os.path.exists(output_dir + base_name + "0.o"):
                        print(f"Copy {base_name + '0.o'} -> {fname + '.o'}")
                        shutil.copy(output_dir + base_name + "0.o", output_dir + fname + ".o")
                        continue
                    elif os.path.exists(output_dir + base_name + "1.o"):
                        print(f"Copy {base_name + '1.o'} -> {fname + '.o'}")
                        shutil.copy(output_dir + base_name + "1.o", output_dir + fname + ".o")
                        continue
                else:
                    cmd = f"{compiler_path} {testcase_dir + src} -O3 -o {output_dir + fname}.o"
                    if a_option:
                        cmd += " -a"
                    cp = subprocess.run(cmd, shell=True, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL)
                    if cp.returncode != 0:
                        record[src] = {"retval": cp.returncode, "err_detail": cp.stderr}
                    else:
                        record[src] = {"retval": 0}
                    if base_name not in {"stencil", "large_loop_array_", "recursive_call_"}:
                        compiled_files.add(base_name)
                    print(src, record[src])
        else:
            print(testcase_dir, " not exist")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', action='store_true', help='Add -a option to compiler command')
    args = parser.parse_args()
    
    compile_object(a_option=args.a)