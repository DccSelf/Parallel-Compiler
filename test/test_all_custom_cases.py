import os
import argparse
from test_pair_custom_cases import compare_single_case

def run_and_compare(a_option=False):
    print("\n{:<20} {:<8} {:<15} {:<15} {:<10}".format(
        "TestCase", "Status", "TestSysy(s)", "TestTensor(s)", "Comparison"
    ))
    print("-" * 80)

    total_execution_time_sysy = 0
    total_execution_time_tensor = 0
    score = 0
    total_cases = 0

    testcase_dir = "./testcase"
    output_o_dir = "./output_o"
    output_dir = "./output_out"
    os.makedirs(output_dir, exist_ok=True)

    sysy_files = sorted([f for f in os.listdir(os.path.join(output_o_dir, "TestSysy")) if f.endswith(".o")])
    tensor_files = set([f for f in os.listdir(os.path.join(output_o_dir, "TestTensor")) if f.endswith(".o")])
    common_files = [f for f in sysy_files if f in tensor_files]

    for src in common_files:
        total_cases += 1
        result = compare_single_case(src, testcase_dir, output_o_dir, output_dir, a_option)
        print(f"{result['name']:<20} {result['status']:<8} {result['TestSysy_time']:<15.6f} {result['TestTensor_time']:<15.6f} {result['compare_result']:<10}")
        
        total_execution_time_sysy += result['TestSysy_time']
        total_execution_time_tensor += result['TestTensor_time']
        
        if result['compare_result'] == 'ok':
            score += 1

    print("-" * 80)
    print("{:<20} {:<8} {:<15.6f} {:<15.6f} {:<10}".format(
        "Total", "",
        total_execution_time_sysy,
        total_execution_time_tensor,
        f"{score}/{total_cases}"
    ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run and compare test results for all common source files.")
    parser.add_argument('-a', action='store_true', help='Add --target=armv7-unknown-linux-gnueabihf option to clang command')
    args = parser.parse_args()

    run_and_compare(a_option=args.a)