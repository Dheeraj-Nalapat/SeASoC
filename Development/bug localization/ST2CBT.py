import ast

class Node:
    def __init__(self, value, node_type):
        self.value = value
        self.node_type = node_type
        self.children = []

# ST2CBT Algorithm
def st_to_cbt(node):
    if not node.children:
        return node

    # Recursively convert children
    new_children = [st_to_cbt(child) for child in node.children]

    # Ensure binary tree structure
    if len(new_children) > 2 or len(new_children) == 1:
        new_node = Node("NewNode", "Type4")
        new_node.children = new_children
        return new_node
    else:
        return node

# Function to split AST into statement tree based on rules
def split_to_statement_tree(node):
    statement_tree = Node("Root", "Type0")

    if isinstance(node, ast.ClassDef):
        # Type1: Class declaration
        head = Node(node.name, "Type1")
        body = [split_to_statement_tree(stmt) for stmt in node.body]
        statement_tree.children.append(Node("head", "Type1"))
        statement_tree.children.append(head)
        statement_tree.children.append(Node("body", "Type1"))
        statement_tree.children.extend(body)

    elif isinstance(node, ast.If):
        # Type2: Condition statement
        control = split_to_statement_tree(node.test)
        block = [split_to_statement_tree(stmt) for stmt in node.body]
        statement_tree.children.append(Node("control", "Type2"))
        statement_tree.children.append(control)
        statement_tree.children.append(Node("block", "Type2"))
        statement_tree.children.extend(block)

    elif isinstance(node, ast.Try):
        # Type3: Try-catch statement
        block = [split_to_statement_tree(stmt) for stmt in node.body]
        catches = [split_to_statement_tree(ex) for ex in node.handlers]
        statement_tree.children.append(Node("block", "Type3"))
        statement_tree.children.extend(block)
        statement_tree.children.append(Node("catches", "Type3"))
        statement_tree.children.extend(catches)

    else:
        # Directly convert other statements to statement tree
        statement_tree.children.append(Node(ast.dump(node), "Type0"))

    return statement_tree

# Sample code content
file_path = r'D:\User\Documents\codes\SeASoC\Development\Testing Code\event_planer.py'
# Example code
with open(file_path, 'r') as file:
    source_code = file.read()

# Parse the sample code into AST
ast_tree = ast.parse(source_code)

# Split AST into statement tree
statement_tree = split_to_statement_tree(ast_tree)

# Convert statement tree to CBT
cbt_root = st_to_cbt(statement_tree)

# Print the CBT
print(cbt_root.value)
print(cbt_root.node_type)
for child in cbt_root.children:
    print(child.value, child.node_type)
    for sub_child in child.children:
        print(sub_child.value, sub_child.node_type)