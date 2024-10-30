import os
import subprocess
import shutil
import re
import argparse

def run_and_compare(a_option=False):
    print("\n{:<25} {:<8} {:<15} {:<10}".format("TestCase", "Status", "Time(s)", "Comparison"))
    print("-" * 55)

    testcase = "./testcase/"
    output_o_case = "./output_o/"
    output_case = "./output_out/"
    total_score = 0

    # 各目录的测试样例数量
    case_counts = {
        "functional": 100,
        "hidden_functional": 40,
        "performance": 59,
        "final_performance": 36
    }

    # 计算总测试样例数量
    total_cases = sum(case_counts.values())

    if not os.path.exists(output_case):
        os.mkdir(output_case)

    for i in case_counts.keys():
        testcase_dir = testcase + i + '/'
        output_o_dir = output_o_case + i + '/'
        output_dir = output_case + i + '/'
        
        if not os.path.exists(output_o_dir):
            print(f"{output_o_dir} not exist, skipping...")
            continue

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)
        
        total_execution_time = 0
        files = sorted([f for f in os.listdir(output_o_dir) if f.endswith(".o")])

        for src in files:
            fname = src.split('.')[0]

            OUTPUT_OBJ = output_o_dir + src
            OUTPUT_EXEC = output_dir + fname
            LIB_NAME = "sysy"
            RESULT_FILE = output_dir + fname + ".out"
            execution_time = 0.0

            # Linking
            cmd_link = f'clang {OUTPUT_OBJ} -L. -l{LIB_NAME} -o {OUTPUT_EXEC} -L/opt/llvm-17/lib -lomp'
            if a_option:
                cmd_link += ' --target=armv7-unknown-linux-gnueabihf'
            #print(cmd_link)
            cp2 = subprocess.run(cmd_link, shell=True, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL)
            if cp2.returncode != 0:
                print("{:<25} {:<8} {:<15} {:<10}".format(src, "fail", "-", "Link Error"))
                continue

            # Execution
            in_file = testcase_dir + fname + ".in"
            cmd_run = f'{OUTPUT_EXEC} < {in_file}' if os.path.exists(in_file) else f'{OUTPUT_EXEC}'
            cp = subprocess.run(cmd_run, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            with open(RESULT_FILE, 'w') as result_file:
                if cp.stdout:
                    result_file.write(cp.stdout.decode('utf-8').strip() + '\n')
                if cp.stderr:
                    line = cp.stderr.decode('utf-8')
                    match = re.search(r'TOTAL: (\d+)H-(\d+)M-(\d+)S-(\d+)us', line)
                    if match:
                        hours, minutes, seconds, us_seconds = map(int, match.groups())
                        execution_time = (hours * 3600 + minutes * 60 + seconds) + us_seconds / 1000000.0
                result_file.write(f"{cp.returncode}\n")

            # Comparison
            cmd_diff = f'diff {testcase_dir}{fname}.out {RESULT_FILE} -wB'
            cp_diff = subprocess.run(cmd_diff, shell=True, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL)
            
            comparison_result = "ok" if cp_diff.returncode == 0 else "fail"
            if comparison_result == "ok":
                total_score += 1

            print("{:<25} {:<8} {:<15.4f} {:<10}".format(src, "ok", execution_time, comparison_result))
            total_execution_time += execution_time

            # Remove .out file
            os.remove(RESULT_FILE)

        print("{:<25} {:<8} {:<15.4f}".format(f"Total Time ({i}):", "", total_execution_time))

    print(f"\nFinal Score: {total_score}/{total_cases}")
    print(f"Percentage: {(total_score/total_cases)*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', action='store_true', help='Add --target=armv7-unknown-linux-gnueabihf option to clang command')
    args = parser.parse_args()
    
    run_and_compare(a_option=args.a)