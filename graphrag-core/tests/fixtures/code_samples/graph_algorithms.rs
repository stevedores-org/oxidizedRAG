use std::collections::{HashMap, HashSet, VecDeque, BinaryHeap};
use std::cmp::Reverse;

/// A node identifier in the graph.
pub type NodeId = usize;

/// A weighted, directed graph represented as an adjacency list.
pub struct Graph {
    /// Adjacency list: node -> [(neighbor, weight)]
    adjacency: HashMap<NodeId, Vec<(NodeId, u32)>>,
    /// Number of nodes in the graph.
    node_count: usize,
}

impl Graph {
    /// Create a new empty graph.
    pub fn new(node_count: usize) -> Self {
        Graph {
            adjacency: HashMap::new(),
            node_count,
        }
    }

    /// Add a directed edge from `src` to `dst` with the given weight.
    pub fn add_edge(&mut self, src: NodeId, dst: NodeId, weight: u32) {
        self.adjacency
            .entry(src)
            .or_insert_with(Vec::new)
            .push((dst, weight));
    }

    /// Get the neighbors of a node.
    pub fn neighbors(&self, node: NodeId) -> &[(NodeId, u32)] {
        self.adjacency.get(&node).map_or(&[], |v| v.as_slice())
    }

    /// Get the total number of nodes.
    pub fn node_count(&self) -> usize {
        self.node_count
    }
}

/// Perform a breadth-first search starting from the given node.
///
/// Returns nodes in BFS traversal order. Explores nodes level by level,
/// visiting all neighbors at the current depth before moving deeper.
pub fn bfs(graph: &Graph, start: NodeId) -> Vec<NodeId> {
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    let mut result = Vec::new();

    visited.insert(start);
    queue.push_back(start);

    while let Some(node) = queue.pop_front() {
        result.push(node);

        for &(neighbor, _weight) in graph.neighbors(node) {
            if visited.insert(neighbor) {
                queue.push_back(neighbor);
            }
        }
    }

    result
}

/// Perform a depth-first search starting from the given node.
///
/// Returns nodes in DFS traversal order. Explores as far as possible
/// along each branch before backtracking.
pub fn dfs(graph: &Graph, start: NodeId) -> Vec<NodeId> {
    let mut visited = HashSet::new();
    let mut stack = vec![start];
    let mut result = Vec::new();

    while let Some(node) = stack.pop() {
        if visited.insert(node) {
            result.push(node);

            // Push neighbors in reverse order for consistent traversal
            let neighbors: Vec<_> = graph.neighbors(node).iter().rev().collect();
            for &(neighbor, _weight) in neighbors {
                if !visited.contains(&neighbor) {
                    stack.push(neighbor);
                }
            }
        }
    }

    result
}

/// Compute shortest paths from a starting node using Dijkstra's algorithm.
///
/// Returns a map from each reachable node to its shortest distance from `start`.
/// Uses a min-heap priority queue for O((V + E) log V) complexity.
pub fn dijkstra(graph: &Graph, start: NodeId) -> HashMap<NodeId, u32> {
    let mut distances: HashMap<NodeId, u32> = HashMap::new();
    let mut heap = BinaryHeap::new();

    distances.insert(start, 0);
    heap.push(Reverse((0u32, start)));

    while let Some(Reverse((cost, node))) = heap.pop() {
        // Skip if we already found a shorter path
        if let Some(&best) = distances.get(&node) {
            if cost > best {
                continue;
            }
        }

        for &(neighbor, weight) in graph.neighbors(node) {
            let new_cost = cost + weight;
            let is_shorter = distances
                .get(&neighbor)
                .map_or(true, |&current| new_cost < current);

            if is_shorter {
                distances.insert(neighbor, new_cost);
                heap.push(Reverse((new_cost, neighbor)));
            }
        }
    }

    distances
}

/// Detect if the graph contains a cycle using DFS.
///
/// Returns `true` if any cycle exists in the directed graph.
pub fn has_cycle(graph: &Graph) -> bool {
    let mut visited = HashSet::new();
    let mut in_stack = HashSet::new();

    for node in 0..graph.node_count() {
        if !visited.contains(&node) {
            if dfs_cycle_check(graph, node, &mut visited, &mut in_stack) {
                return true;
            }
        }
    }

    false
}

/// Helper for cycle detection using recursive DFS.
fn dfs_cycle_check(
    graph: &Graph,
    node: NodeId,
    visited: &mut HashSet<NodeId>,
    in_stack: &mut HashSet<NodeId>,
) -> bool {
    visited.insert(node);
    in_stack.insert(node);

    for &(neighbor, _weight) in graph.neighbors(node) {
        if !visited.contains(&neighbor) {
            if dfs_cycle_check(graph, neighbor, visited, in_stack) {
                return true;
            }
        } else if in_stack.contains(&neighbor) {
            return true;
        }
    }

    in_stack.remove(&node);
    false
}

/// Compute a topological ordering of the graph.
///
/// Returns `None` if the graph contains a cycle.
/// Uses Kahn's algorithm with in-degree tracking.
pub fn topological_sort(graph: &Graph) -> Option<Vec<NodeId>> {
    let mut in_degree: HashMap<NodeId, usize> = HashMap::new();

    // Initialize in-degrees
    for node in 0..graph.node_count() {
        in_degree.entry(node).or_insert(0);
        for &(neighbor, _) in graph.neighbors(node) {
            *in_degree.entry(neighbor).or_insert(0) += 1;
        }
    }

    // Start with nodes that have zero in-degree
    let mut queue: VecDeque<NodeId> = in_degree
        .iter()
        .filter(|(_, &deg)| deg == 0)
        .map(|(&node, _)| node)
        .collect();

    let mut result = Vec::new();

    while let Some(node) = queue.pop_front() {
        result.push(node);

        for &(neighbor, _) in graph.neighbors(node) {
            if let Some(deg) = in_degree.get_mut(&neighbor) {
                *deg -= 1;
                if *deg == 0 {
                    queue.push_back(neighbor);
                }
            }
        }
    }

    if result.len() == graph.node_count() {
        Some(result)
    } else {
        None // Graph has a cycle
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_graph() -> Graph {
        let mut g = Graph::new(5);
        g.add_edge(0, 1, 4);
        g.add_edge(0, 2, 1);
        g.add_edge(2, 1, 2);
        g.add_edge(1, 3, 1);
        g.add_edge(2, 3, 5);
        g.add_edge(3, 4, 3);
        g
    }

    #[test]
    fn test_bfs_traversal() {
        let g = sample_graph();
        let order = bfs(&g, 0);
        assert_eq!(order[0], 0);
        assert!(order.contains(&1));
        assert!(order.contains(&2));
    }

    #[test]
    fn test_dijkstra_shortest_paths() {
        let g = sample_graph();
        let distances = dijkstra(&g, 0);
        assert_eq!(distances[&0], 0);
        assert_eq!(distances[&1], 3); // 0->2->1 = 1+2
        assert_eq!(distances[&2], 1); // 0->2 = 1
    }
}
