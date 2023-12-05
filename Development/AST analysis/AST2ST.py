import ast

class StatementTreeVisitor(ast.NodeVisitor):
    def __init__(self):
        self.statement_trees = []

    def visit_FunctionDef(self, node):
        # For each function definition, create a new statement tree
        current_tree = []
        current_tree.append(ast.dump(node))
        self.statement_trees.append(current_tree)

        # Continue processing the function body
        self.generic_visit(node)

    def visit_Expr(self, node):
        # For each expression (statement), add it to the current statement tree
        current_tree = self.statement_trees[-1]
        current_tree.append(ast.dump(node))

    def generic_visit(self, node):
        # Override the generic visit method to handle other nodes
        if isinstance(node, ast.stmt):
            # For other statements, add them to the current statement tree
            current_tree = self.statement_trees[-1]
            current_tree.append(ast.dump(node))

        # Continue processing child nodes
        super().generic_visit(node)

# Example code
file_path = r'D:\User\Documents\codes\SeASoC\Development\Testing Code\event_planer.py'
# Example code
with open(file_path, 'r') as file:
    source_code = file.read()

# Parse the example code
ast_tree = ast.parse(source_code)

# Create an instance of the custom visitor
visitor = StatementTreeVisitor()

# Traverse the AST with the visitor
visitor.visit(ast_tree)

# Print the statement trees
for i, tree in enumerate(visitor.statement_trees, start=1):
    print(f"Statement Tree {i}:")
    for statement in tree:
        print(statement)
    print()
