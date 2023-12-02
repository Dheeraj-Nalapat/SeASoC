import ast

class MyVisitor(ast.NodeVisitor):
    def visit_FunctionDef(self, node):
        print(f"Function Definition: {node.name}")
        self.generic_visit(node)

    def visit_Call(self, node):
        print(f"Function Call: {node.func.id}")
        self.generic_visit(node)

    def visit_Name(self, node):
        print(f"Variable or Function Name: {node.id}")
        self.generic_visit(node)

# Example code
sample_code = """
def greet(name):
    print(f"Hello, {name}!")

greet("World")
"""

# Generate AST for the example code
ast_tree = ast.parse(sample_code)

# Create an instance of the custom visitor
visitor = MyVisitor()

# Traverse the AST with the visitor
visitor.visit(ast_tree)
