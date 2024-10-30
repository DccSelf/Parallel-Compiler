import os
import subprocess
import argparse
from test_single_custom_case import run_single_test

def compare_single_case(src, testcase_dir="./testcase", output_o_dir="./output_o", output_dir="./output_out", a_option=False):
    results = {}
    for test_type in ["TestSysy", "TestTensor"]:
        results[test_type] = run_single_test(src, test_type, testcase_dir, output_o_dir, output_dir, a_option)

    # Compare results
    compare_result = "fail"
    if results['TestSysy']['status'] == 'ok' and results['TestTensor']['status'] == 'ok':
        cmd = f'diff {results["TestSysy"]["file"]} {results["TestTensor"]["file"]} -wB'
        cp_diff = subprocess.run(cmd, shell=True, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL)
        compare_result = "ok" if cp_diff.returncode == 0 else "fail"

    # Clean up
    for test_type in ["TestSysy", "TestTensor"]:
        if os.path.exists(results[test_type]['file']):
            os.remove(results[test_type]['file'])
        exec_file = os.path.join(output_dir, test_type, src.split('.')[0])
        if os.path.exists(exec_file):
            os.remove(exec_file)

    return {
        'name': src,
        'status': 'ok',
        'TestSysy_time': results['TestSysy']['time'],
        'TestTensor_time': results['TestTensor']['time'],
        'compare_result': compare_result
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare test results for a specific source file.")
    parser.add_argument("src", help="Source file name")
    parser.add_argument("--testcase_dir", default="./testcase", help="Directory containing test cases")
    parser.add_argument("--output_o_dir", default="./output_o", help="Directory for output object files")
    parser.add_argument("--output_dir", default="./output_out", help="Directory for output results")
    parser.add_argument('-a', action='store_true', help='Add --target=armv7-unknown-linux-gnueabihf option to clang command')
    args = parser.parse_args()

    result = compare_single_case(args.src, args.testcase_dir, args.output_o_dir, args.output_dir, args.a)
    print(f"Comparison result for {result['name']}:")
    print(f"TestSysy time: {result['TestSysy_time']} seconds")
    print(f"TestTensor time: {result['TestTensor_time']} seconds")
    print(f"Comparison result: {result['compare_result']}")