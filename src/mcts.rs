use rand::distr::weighted::WeightedIndex;
use rand::distr::Distribution;
use rand_distr::multi::Dirichlet;
use std::fmt::{self, Display};
use std::ops::{Index, IndexMut};
use tracing::{debug, enabled, trace, Level};

use crate::evaluator::*;
use crate::games::{Action, GameState};
use crate::util::normalize;

// PUCT exploration constant. Typically 1.0 to 2.5.
pub(crate) const C_PUCT: f32 = 1.0;

// Searches from `g` and returns the most-visited move and the position's
// value for the side to move. No root noise.
pub fn best_move<G: GameState>(
    g: &G,
    eval: &mut impl Evaluator<G>,
    num_rollouts: usize,
) -> (Action, f32) {
    let dhp = DirichletHParams::default();
    let (mcts_post, val_post) = mcts(g, eval, num_rollouts, &dhp);
    let action = select_action(&mcts_post, true);
    debug!("Current eval: {}", val_post);
    return (action, val_post);
}

// Pick an action from a probability distribution over actions.
// greedy=true takes the most likely action (temperature -> 0),
// otherwise samples proportionally (temperature 1).
pub fn select_action(post_prob: &[f32], greedy: bool) -> Action {
    if greedy {
        let mut best = 0;
        for (i, &p) in post_prob.iter().enumerate() {
            if p > post_prob[best] {
                best = i;
            }
        }
        return best;
    }
    return WeightedIndex::new(post_prob)
        .unwrap()
        .sample(&mut rand::rng());
}

// Run MCTS from the current state with a fresh tree. State must be
// non-terminal. num sims >= 1.
// Returns (posterior, value): the posterior is a probability distribution
// over actions (share of root visits), the value is the root's mean value
// from the side-to-move's perspective.
pub fn mcts<G: GameState>(
    state: &G,
    eval: &mut impl Evaluator<G>,
    num_sims: usize,
    dhp: &DirichletHParams,
) -> (Vec<f32>, f32) {
    return search(&mut MctsTree::new(C_PUCT), state, eval, num_sims, dhp);
}

// Runs MCTS on `tree`, whose root must correspond to `state` (non-terminal),
// until the root has num_sims visits. Visits already in the tree (from a
// reused subtree) count toward num_sims; an unexpanded root always gets at
// least the one simulation that expands it. Dirichlet noise is mixed into the
// root's priors before searching. Returns (posterior, value) as for mcts().
pub(crate) fn search<G: GameState>(
    tree: &mut MctsTree,
    state: &G,
    eval: &mut impl Evaluator<G>,
    num_sims: usize,
    dhp: &DirichletHParams,
) -> (Vec<f32>, f32) {
    if !tree[0].is_expanded() {
        mcts_sim(tree, state, eval); // expands the root
    }
    tree.add_root_noise(dhp);
    let n_new = num_sims.saturating_sub(tree[0].visits as usize);
    trace!(
        "Root has {} visits, running {} more simulations",
        tree[0].visits,
        n_new
    );
    for _ in 0..n_new {
        mcts_sim(tree, state, eval);
    }
    let root = &tree[0];
    let mcts_post = mcts_tree_posterior(tree, 0, G::ACTN_SPC_SIZE);
    let val_post = root.value / root.visits as f32;
    if enabled!(Level::DEBUG) {
        let mcts_pri = mcts_tree_prior(tree, 0, G::ACTN_SPC_SIZE);
        debug!("Mcts root node: {}", root);
        debug!(
            "total probs should be 1.0. mcts_pri: {}  mcts_post: {}",
            mcts_pri.iter().sum::<f32>(),
            mcts_post.iter().sum::<f32>()
        );
        debug!("MCTS prior: {:?}", &mcts_pri);
        debug!("MCTS posterior: {:?}", &mcts_post);
        debug!("MCTS posterior value: {}", val_post);
    }
    return (mcts_post, val_post);
}

// Performs a single MCTS simulation using PUCT
fn mcts_sim<G: GameState>(tree: &mut MctsTree, state_orig: &G, eval: &mut impl Evaluator<G>) {
    let mut state = state_orig.clone();
    let mut node_id = 0; // Root
                         // Drill down to a leaf node.
    while tree[node_id].is_expanded() {
        let (child_id, action) = tree.puct_select(node_id);
        node_id = child_id;
        state.apply_action(action);
    }
    // Is leaf node (possibly terminal)
    let value = if state.is_terminal() {
        state.get_terminal_value()
    } else {
        tree.expand(node_id, &state, eval)
    };
    tree.back_propagate(node_id, value);
}

// Returns the MCTS posterior: each child's share of the visits to its siblings.
// node_id must be an expanded node. If no child has been visited yet (only one
// simulation ran) it falls back to the (noised) priors.
fn mcts_tree_posterior(tree: &MctsTree, node_id: NodeId, actn_spc_size: usize) -> Vec<f32> {
    let mut p = vec![0.0; actn_spc_size];
    for child_id in tree.children(node_id) {
        let child = &tree[child_id];
        p[child.action.unwrap()] = child.visits as f32;
    }
    if p.iter().all(|&x| x == 0.0) {
        return mcts_tree_prior(tree, node_id, actn_spc_size);
    }
    return normalize(p);
}

// Returns the children's prior probabilities (0 for actions with no child).
fn mcts_tree_prior(tree: &MctsTree, node_id: NodeId, actn_spc_size: usize) -> Vec<f32> {
    let mut p = vec![0.0; actn_spc_size];
    for child_id in tree.children(node_id) {
        let child = &tree[child_id];
        p[child.action.unwrap()] = child.prior;
    }
    return p;
}

type NodeId = usize;

pub struct MctsNode {
    pub parent: Option<NodeId>, // Parent index for backpropagation. None for root node.
    pub first_child: Option<NodeId>, // First element in the children linked list.
    // None if node is terminal or not yet expanded.
    pub next_sibling: Option<NodeId>, // Next sibling index.
    pub action: Option<Action>,       // The action that led to this state
    pub prior: f32,  // The prior probability of selecting this node from the parent.
    pub visits: u32, // Number of times this node has been visited. 1 + Sum(child_visits)
    pub value: f32,  // Total value for this node.
                     // If not expanded, is 0.0.
                     // If expanded, value - sum(child_values).
}

impl MctsNode {
    pub fn is_expanded(&self) -> bool {
        return self.first_child.is_some();
    }
}

impl fmt::Display for MctsNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut avg_val = 0.0;
        if self.visits > 0 {
            avg_val = self.value / (self.visits as f32);
        }
        write!(
            f,
            "visits: {}   avg_value: {}   is_expanded: {}",
            self.visits,
            avg_val,
            self.is_expanded()
        )?;
        return Ok(());
    }
}

pub struct Children<'a> {
    tree: &'a MctsTree,
    next_id: Option<NodeId>,
}

impl Iterator for Children<'_> {
    type Item = NodeId;
    fn next(&mut self) -> Option<Self::Item> {
        let current_id = self.next_id?;
        self.next_id = self.tree[current_id].next_sibling;
        return Some(current_id);
    }
}

// Memory-pool implementation of MCTS Tree
// Root node has NodeId == 0;
pub struct MctsTree {
    nodes_: Vec<MctsNode>,
    c_puct: f32,
}

impl MctsTree {
    pub fn new(c_puct: f32) -> Self {
        let mut x = Self {
            nodes_: Vec::new(),
            c_puct,
        };
        let root = MctsNode {
            parent: None,
            first_child: None,
            next_sibling: None,
            action: None,
            prior: 1.0,
            visits: 0,
            value: 0.0,
        };
        x.nodes_.push(root);
        return x;
    }

    // Returns a new tree rooted at the root's child for `action`, keeping that
    // child's statistics and its whole subtree. Returns a fresh tree if the
    // root has no child for `action` (e.g. the root was never expanded).
    pub fn reroot(&self, action: Action) -> MctsTree {
        let Some(new_root) = self.children(0).find(|&id| self[id].action == Some(action)) else {
            return MctsTree::new(self.c_puct);
        };
        let mut tree = MctsTree {
            nodes_: Vec::new(),
            c_puct: self.c_puct,
        };
        // Depth-first copy; the new root is pushed first, so it gets NodeId 0.
        let mut stack: Vec<(NodeId, Option<NodeId>)> = vec![(new_root, None)];
        while let Some((old_id, new_parent)) = stack.pop() {
            let old = &self[old_id];
            let new_id = tree.nodes_.len();
            tree.nodes_.push(MctsNode {
                parent: new_parent,
                first_child: None,
                next_sibling: new_parent.and_then(|p| tree[p].first_child),
                action: if new_parent.is_some() {
                    old.action
                } else {
                    None
                },
                prior: old.prior,
                visits: old.visits,
                value: old.value,
            });
            if let Some(p) = new_parent {
                tree[p].first_child = Some(new_id);
            }
            for child_id in self.children(old_id) {
                stack.push((child_id, Some(new_id)));
            }
        }
        return tree;
    }

    fn children(&self, node_id: NodeId) -> Children<'_> {
        return Children {
            tree: self,
            next_id: self[node_id].first_child,
        };
    }

    // Adds value to node_id and every ancestor, flipping sign each level
    // because each level is from the other player's perspective.
    fn back_propagate(&mut self, node_id: NodeId, value: f32) {
        let mut curr_id = Some(node_id);
        let mut polarity = 1.0;
        while let Some(id) = curr_id {
            let curr = &mut self[id];
            curr.value += value * polarity;
            curr.visits += 1;
            polarity *= -1.0;
            curr_id = curr.parent;
        }
    }

    // Returns the child of node_id with the highest PUCT score
    // (probabilistic upper confidence bound applied to trees), and its action.
    // Panics if node_id has no children.
    fn puct_select(&self, node_id: NodeId) -> (NodeId, Action) {
        let (best_child_id, _) = self
            .children(node_id)
            .map(|id| (id, self.calc_puct(id)))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .expect("puct_select called on a node with no children");
        return (best_child_id, self[best_child_id].action.unwrap());
    }

    // Calculates PUCT score for a given node relative to its parent.
    // Node must have a parent.
    fn calc_puct(&self, node_id: NodeId) -> f32 {
        let node = &self[node_id];
        let parent = &self[node.parent.unwrap()];
        let exploitation_term = if node.visits == 0 {
            0.0
        } else {
            -node.value / (node.visits as f32)
        };
        let exploration_term =
            self.c_puct * node.prior * (parent.visits as f32).sqrt() / ((1 + node.visits) as f32);
        return exploitation_term + exploration_term;
    }

    // Must not be called on a terminal state (debug-asserted; in release it
    // would evaluate a finished game and create children for it).
    // Returns the value of the node
    fn expand<G: GameState>(
        &mut self,
        node_id: NodeId,
        state: &G,
        eval: &mut impl Evaluator<G>,
    ) -> f32 {
        debug_assert!(!state.is_terminal());
        trace!("Expanding node {}", node_id);
        let (p, v) = eval.evaluate(state);
        let legal_mask = state.legal_actions_mask();
        for action in 0..p.len() {
            // Every legal action gets a child, even if its prior underflowed to 0.
            if legal_mask[action] {
                self.add_child(action, node_id, p[action]);
            }
        }
        return v;
    }

    // Mixes Dirichlet noise into the priors of the root's children (which are
    // exactly the legal moves). The priors already sum to 1, so they still do.
    // No-op if noise is disabled or there are fewer than 2 children
    // (Dirichlet needs at least 2 components).
    fn add_root_noise(&mut self, dhp: &DirichletHParams) {
        if dhp.epsilon == 0.0 {
            return;
        }
        let children: Vec<NodeId> = self.children(0).collect();
        if children.len() < 2 {
            return;
        }
        trace!("Adding dirichlet noise {} to root", dhp);
        let alpha_vec = vec![dhp.alpha; children.len()];
        let noise = Dirichlet::new(&alpha_vec).unwrap().sample(&mut rand::rng());
        for (&id, &n) in children.iter().zip(noise.iter()) {
            let child = &mut self[id];
            child.prior = (1.0 - dhp.epsilon) * child.prior + dhp.epsilon * n;
        }
    }

    // Create a new node in the arena-pool. Return it's id.
    fn add_child(&mut self, action: Action, parent: NodeId, prior: f32) -> NodeId {
        let child_id: NodeId = self.nodes_.len();
        let child = MctsNode {
            action: Some(action),
            parent: Some(parent),
            first_child: None,
            next_sibling: self.nodes_[parent].first_child,
            prior,
            visits: 0,
            value: 0.0,
        };
        self.nodes_.push(child);
        self.nodes_[parent].first_child = Some(child_id);
        return child_id;
    }
}

impl Index<NodeId> for MctsTree {
    type Output = MctsNode;
    fn index(&self, node_id: NodeId) -> &MctsNode {
        return &self.nodes_[node_id];
    }
}

impl IndexMut<NodeId> for MctsTree {
    fn index_mut(&mut self, node_id: NodeId) -> &mut MctsNode {
        return &mut self.nodes_[node_id];
    }
}

#[derive(Debug, Clone)]
pub struct DirichletHParams {
    pub epsilon: f32, // [0,1]. Mixing param. What percent of prior is from dirichlet noise
    // Set to 0 to disable dirichlet noise.
    pub alpha: f32, // Concentration parameter.
}

impl Default for DirichletHParams {
    fn default() -> Self {
        return Self {
            epsilon: 0.0,
            alpha: 1.0,
        };
    }
}

impl Display for DirichletHParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        return write!(
            f,
            "DirichletHParams{{epsilon: {}, alpha: {} }}",
            self.epsilon, self.alpha
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::minipente::{self, MiniPenteState};

    // Uniform-policy, zero-value evaluator that counts its calls.
    struct Counting {
        n: usize,
    }
    impl Evaluator<MiniPenteState> for Counting {
        fn evaluate(&mut self, g: &MiniPenteState) -> (Vec<f32>, f32) {
            self.n += 1;
            return NullEvaluator {}.evaluate(g);
        }
    }

    fn noise() -> DirichletHParams {
        return DirichletHParams {
            epsilon: 0.25,
            alpha: 0.25,
        };
    }

    fn position(moves: &[(usize, usize)]) -> MiniPenteState {
        let mut g = MiniPenteState::new();
        for &(r, c) in moves {
            assert!(g.play(r, c));
        }
        return g;
    }

    fn assert_distribution(p: &[f32]) {
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "sums to {sum}");
        assert!(p.iter().all(|&x| (0.0..=1.0).contains(&x)));
    }

    // Number of nodes in the subtree rooted at node_id (inclusive).
    fn subtree_size(tree: &MctsTree, node_id: NodeId) -> usize {
        return 1 + tree
            .children(node_id)
            .map(|c| subtree_size(tree, c))
            .sum::<usize>();
    }

    // Structural invariants: parent links match, and every expanded node has
    // exactly 1 + (sum of its children's visits) visits.
    fn assert_consistent(tree: &MctsTree) {
        assert!(tree[0].parent.is_none());
        for id in 0..tree.nodes_.len() {
            let child_visits: u32 = tree.children(id).map(|c| tree[c].visits).sum();
            for c in tree.children(id) {
                assert_eq!(tree[c].parent, Some(id));
            }
            if tree[id].is_expanded() {
                assert_eq!(tree[id].visits, 1 + child_visits, "node {id}");
            }
        }
    }

    #[test]
    fn posterior_is_a_distribution_over_legal_moves() {
        let g = position(&[(4, 4), (4, 5), (3, 3)]);
        let (post, value) = mcts(&g, &mut NullEvaluator {}, 100, &noise());
        assert_distribution(&post);
        for (a, &legal) in g.legal_actions_mask().iter().enumerate() {
            if !legal {
                assert_eq!(post[a], 0.0, "illegal action {a} has mass");
            }
        }
        assert!((-1.0..=1.0).contains(&value));
    }

    #[test]
    fn single_simulation_posterior_falls_back_to_priors() {
        let g = position(&[(4, 4)]);
        let (post, _) = mcts(&g, &mut NullEvaluator {}, 1, &DirichletHParams::default());
        let uniform = 1.0 / 80.0;
        for (a, &legal) in g.legal_actions_mask().iter().enumerate() {
            let expected = if legal { uniform } else { 0.0 };
            assert!((post[a] - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn fresh_search_runs_exactly_num_sims_evaluations() {
        // No terminal positions are reachable this early, so every
        // simulation ends in one network evaluation.
        let mut ev = Counting { n: 0 };
        mcts(&MiniPenteState::new(), &mut ev, 200, &noise());
        assert_eq!(ev.n, 200);
    }

    #[test]
    fn finds_an_immediate_win() {
        // Black has four in a row on row 0 and is to move.
        let g = position(&[
            (0, 0),
            (8, 0),
            (0, 1),
            (8, 2),
            (0, 2),
            (8, 4),
            (0, 3),
            (8, 6),
        ]);
        assert!(g.is_black());
        let (post, _) = mcts(&g, &mut NullEvaluator {}, 400, &DirichletHParams::default());
        let win = minipente::coord_2_action(0, 4);
        assert_eq!(select_action(&post, true), win);
        assert!(
            post[win] > 0.5,
            "winning move got only {} of the visits",
            post[win]
        );
    }

    #[test]
    fn back_propagate_alternates_sign_up_the_tree() {
        let mut tree = MctsTree::new(C_PUCT);
        let child = tree.add_child(0, 0, 1.0);
        let grandchild = tree.add_child(1, child, 1.0);
        tree.back_propagate(grandchild, 1.0);
        assert_eq!(tree[grandchild].value, 1.0);
        assert_eq!(tree[child].value, -1.0);
        assert_eq!(tree[0].value, 1.0);
        for id in [0, child, grandchild] {
            assert_eq!(tree[id].visits, 1);
        }
    }

    #[test]
    fn reroot_keeps_the_chosen_subtree_and_its_statistics() {
        let g = MiniPenteState::new();
        let mut tree = MctsTree::new(C_PUCT);
        let (post, _) = search(&mut tree, &g, &mut NullEvaluator {}, 300, &noise());
        assert_consistent(&tree);
        let action = select_action(&post, true);
        let old_child = tree
            .children(0)
            .find(|&c| tree[c].action == Some(action))
            .unwrap();
        let (visits, value) = (tree[old_child].visits, tree[old_child].value);
        let size = subtree_size(&tree, old_child);

        let new_tree = tree.reroot(action);
        assert_eq!(new_tree.nodes_.len(), size);
        assert_eq!(new_tree[0].visits, visits);
        assert_eq!(new_tree[0].value, value);
        assert_eq!(new_tree[0].action, None);
        assert_consistent(&new_tree);
    }

    #[test]
    fn reroot_without_a_matching_child_gives_a_fresh_tree() {
        let tree = MctsTree::new(C_PUCT).reroot(5);
        assert_eq!(tree.nodes_.len(), 1);
        assert_eq!(tree[0].visits, 0);
        assert!(!tree[0].is_expanded());
    }

    #[test]
    fn reused_search_only_tops_up_to_num_sims() {
        let mut g = MiniPenteState::new();
        let mut tree = MctsTree::new(C_PUCT);
        let (post, _) = search(&mut tree, &g, &mut NullEvaluator {}, 200, &noise());
        let action = select_action(&post, true);
        g.apply_action(action);
        tree = tree.reroot(action);
        let reused = tree[0].visits as usize;
        assert!(reused > 0 && reused < 200);

        let mut ev = Counting { n: 0 };
        search(&mut tree, &g, &mut ev, 200, &noise());
        assert_eq!(ev.n, 200 - reused);
        assert_eq!(tree[0].visits, 200);
        assert_consistent(&tree);
    }

    #[test]
    fn root_noise_keeps_priors_normalized() {
        let g = MiniPenteState::new();
        let mut tree = MctsTree::new(C_PUCT);
        mcts_sim(&mut tree, &g, &mut NullEvaluator {}); // expand the root
        let priors = |t: &MctsTree| mcts_tree_prior(t, 0, MiniPenteState::ACTN_SPC_SIZE);
        let before = priors(&tree);

        tree.add_root_noise(&DirichletHParams::default()); // epsilon 0: no-op
        assert_eq!(priors(&tree), before);

        tree.add_root_noise(&noise());
        let after = priors(&tree);
        assert_distribution(&after);
        assert_ne!(after, before);
    }

    #[test]
    fn select_action_greedy_takes_argmax_and_sampling_avoids_zero_mass() {
        let p = [0.0, 0.2, 0.0, 0.8];
        assert_eq!(select_action(&p, true), 3);
        for _ in 0..200 {
            let a = select_action(&p, false);
            assert!(a == 1 || a == 3);
        }
    }
}
