import ast

# Sample code content
sample_code = """
class ExampleClass:
    def __init__(self):
        self.value = 0

    def example_method(self, x):
        if x > 0:
            print("Positive")
        else:
            print("Non-positive")
"""

# Parse the sample code into AST
ast_tree = ast.parse(sample_code)

# Function to split AST into statement tree based on rules
def split_to_statement_tree(node):
    statement_tree = []

    if isinstance(node, ast.ClassDef):
        # Type1: Class declaration
        head = [node.name, node.bases, node.keywords]
        body = [split_to_statement_tree(stmt) for stmt in node.body]
        statement_tree.append({"Type": "Type1", "head": head, "body": body})

    elif isinstance(node, ast.If):
        # Type2: Condition statement
        control = split_to_statement_tree(node.test)
        block = [split_to_statement_tree(stmt) for stmt in node.body]
        statement_tree.append({"Type": "Type2", "control": control, "block": block})

    elif isinstance(node, ast.Try):
        # Type3: Try-catch statement
        block = [split_to_statement_tree(stmt) for stmt in node.body]
        catches = [split_to_statement_tree(ex) for ex in node.handlers]
        statement_tree.append({"Type": "Type3", "block": block, "catches": catches})

    else:
        # Directly convert other statements to statement tree
        statement_tree.append({"Type": "Direct", "content": ast.dump(node)})

    return statement_tree

# Split AST into statement tree
statement_tree = split_to_statement_tree(ast_tree)

# Print the statement tree
print(statement_tree)
