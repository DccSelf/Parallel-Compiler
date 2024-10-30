import os
import subprocess
import re
import argparse

def run_single_test(src, test_type, testcase_dir="./testcase", output_o_dir="./output_o", output_dir="./output_out", a_option=False):
    fname = src.split('.')[0]
    output_o_subdir = os.path.join(output_o_dir, test_type)
    output_subdir = os.path.join(output_dir, test_type)
    os.makedirs(output_subdir, exist_ok=True)
    OUTPUT_OBJ = os.path.join(output_o_subdir, src)
    OUTPUT_EXEC = os.path.join(output_subdir, fname)
    LIB_NAME = "sysy"
    RESULT_FILE = os.path.join(output_subdir, f"{fname}.out")

    cmd_link = f'clang {OUTPUT_OBJ} -L. -l{LIB_NAME} -o {OUTPUT_EXEC} -L/opt/llvm-17/lib -lomp'
    if a_option:
        cmd_link += " --target=armv7-unknown-linux-gnueabihf"
    cp2 = subprocess.run(cmd_link, shell=True, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL)
    if cp2.returncode != 0:
        return {'status': 'fail', 'time': 'N/A'}

    in_file = os.path.join(testcase_dir, test_type, f"{fname}.in")
    cmd_run = f'{OUTPUT_EXEC} < {in_file}' if os.path.exists(in_file) else f'{OUTPUT_EXEC}'
    cp = subprocess.run(cmd_run, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    execution_time = 0.0
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

    return {'status': 'ok', 'time': execution_time, 'file': RESULT_FILE}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single test for a specific type.")
    parser.add_argument("src", help="Source file name")
    parser.add_argument("test_type", help="Test type (TestSysy or TestTensor)")
    parser.add_argument("--testcase_dir", default="./testcase", help="Directory containing test cases")
    parser.add_argument("--output_o_dir", default="./output_o", help="Directory for output object files")
    parser.add_argument("--output_dir", default="./output_out", help="Directory for output results")
    parser.add_argument('-a', action='store_true', help='Add --target=armv7-unknown-linux-gnueabihf option to clang command')
    args = parser.parse_args()

    result = run_single_test(args.src, args.test_type, args.testcase_dir, args.output_o_dir, args.output_dir, args.a)
    print(f"Test result for {args.src} ({args.test_type}):")
    print(f"Status: {result['status']}")
    print(f"Execution time: {result['time']} seconds")
    print(f"Result file: {result['file']}")