import ast

class MyVisitor(ast.NodeVisitor):
    def visit_FunctionDef(self, node):
        print(f"Function Definition: {node.name}")
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            print(f"Function Call: {node.func.id}")
        elif isinstance(node.func, ast.Attribute):
            # Handle nested attributes
            attr_chain = []
            current_node = node.func
            while isinstance(current_node, ast.Attribute):
                attr_chain.append(current_node.attr)
                current_node = current_node.value
            if isinstance(current_node, ast.Name):
                attr_chain.append(current_node.id)
                print(f"Method Call: {'.'.join(reversed(attr_chain))}")
        self.generic_visit(node)


    def visit_Name(self, node):
        print(f"Variable or Function Name: {node.id}")

        self.generic_visit(node)

# Example code
file_path = r'D:\User\Documents\codes\SeASoC\Development\Testing Code\event_planer.py'
# Example code
with open(file_path, 'r') as file:
    source_code = file.read()

# Generate AST for the example code
ast_tree = ast.parse(source_code)

# Create an instance of the custom visitor
visitor = MyVisitor()

# Traverse the AST with the visitor
visitor.visit(ast_tree)
