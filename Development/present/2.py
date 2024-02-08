from pylint import lint

def run_pylint(file_path):
    pylint_options = [file_path]
    pylint_output = lint.Run(pylint_options, exit=False)

    # Print available attributes of linter.stats
    print(dir(pylint_output.linter.stats))

    # Access linting results
    if hasattr(pylint_output.linter.stats, 'by_msg'):
        for lint_result in pylint_output.linter.stats.by_msg:
            print(f"{lint_result}: {pylint_output.linter.stats.by_msg[lint_result]} occurrences")
    else:
        print("No linting issues found.")

if __name__ == "__main__":
    python_file_path = 'D:/User/Documents/codes/SeASoC/Development/present/sample.py'
    run_pylint(python_file_path)