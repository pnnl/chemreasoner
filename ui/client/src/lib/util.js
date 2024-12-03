export function traverse(node, handler, depth = 0) {
  handler(node, depth)
  if (node.children) {
    node.children.map((child) => traverse(child, handler, depth + 1))
  }
}