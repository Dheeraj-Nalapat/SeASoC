import ast
import astor

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

file_path = r'D:\User\Documents\codes\SeASoC\Development\Testing Code\event_planer.py'
# Example code
with open(file_path, 'r') as file:
    source_code = file.read()
# (Your sample_code remains the same)

# Generate AST for the example code
ast_tree = ast.parse(source_code)

# Print the AST in tree format using astor
print(astor.dump_tree(ast_tree))
