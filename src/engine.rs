use burn::module::Module;
use burn::tensor::DeviceIndex::Specified;
use burn::tensor::{Device, Tensor, TensorData};
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use tracing::{trace, debug, info};
use std::fmt::Display;
use std::{fmt, vec};
use rand;
use rand_distr::multi::Dirichlet;

use crate::tard::{self, TrainingElement};
use crate::gamestate::{GameState, Action};
use crate::evaluator::*;
use crate::utils::normalize;


pub fn inference <G: GameState>(
    g: &G,
    eval: &mut impl Evaluator<G>,
    num_rollouts: usize) -> (Action, f32)
{   
    let dhp = DirichletHParams::default();
    let (mcts_post, val_post) = mcts(g, eval, num_rollouts, &dhp);
    let mcts_prob = exp(&mcts_post);
    let action = select_action(&mcts_prob, true);
    debug!("Current eval: {}", val_post);
    return (action, val_post);
}


pub fn run_self_play <G: GameState>(
    eval: &mut impl Evaluator<G>,
    num_sims: usize,
    dhp: &DirichletHParams,
    temp_moves: usize,
) -> Vec<SelfPlayElement<G>>{
    debug!("Running self play");
    let mut sp_vec = Vec::new();
    let mut g = G::default(); 
    let mut move_n = 0;
    while !g.is_terminal(){
        debug!("Current State: \n{}", g);
        debug!("Computing self-play move#: {}", move_n);
        let (mcts_post, val_post) = mcts(&g, eval, num_sims, dhp);
        let mcts_prob = exp(&mcts_post);
        // Sample for the first temp_moves plies (opening variety), then play greedily.
        let action = select_action(&mcts_prob, move_n >= temp_moves);
        let sp_element = SelfPlayElement{
            g: g.clone(),
            mcts_post: mcts_post,
            outcome: 0.0
        };
        sp_vec.push(sp_element);
        debug!("Playing action: {}", action);
        g.apply_action(action);
        move_n += 1;
    }
    debug!("Current State: \n{}", g);
    let outcome = g.get_terminal_value() as f32;
    // sp_vec has list of all non-terminal states.
    // outcome is relative to terminal state. So we iterate
    // in reverse and polarity starts at -1.0
    let mut polarity = -1.0;
    for spe in sp_vec.iter_mut().rev(){
        spe.outcome = polarity * outcome;
        polarity *= -1.0;
    }
    trace!("{:?}", &sp_vec);
    return sp_vec;
}

// Pick an action from a probability distribution over actions.
// greedy=true takes the most likely action (temperature -> 0),
// otherwise samples proportionally (temperature 1).
fn select_action(post_prob: &[f32], greedy: bool) -> Action{
    if greedy{
        let mut best = 0;
        for (i, &p) in post_prob.iter().enumerate(){
            if p > post_prob[best]{
                best = i;
            }
        }
        return best;
    }
    return WeightedIndex::new(post_prob)
                         .unwrap()
                         .sample(&mut rand::rng());
}

#[derive(Debug)]
pub struct SelfPlayElement<G:GameState>{
    pub g: G,
    pub mcts_post: Vec<f32>,
    pub outcome: f32
}


// Run MCTS from the current state. State must be non-terminal. num sims >= 1.
// Returns log probabiltiy posterior.
pub fn mcts <G: GameState> (
    state: &G,
    eval: &mut impl Evaluator<G>,
    num_sims: usize,
    dhp: &DirichletHParams,
) -> (Vec<f32>, f32){
    // Run mcts and return the evaluated mcts policy
    let mut tree: MctsTree = MctsTree::new(1.0);
    for _ in 0..num_sims{
        mcts_sim(&mut tree, state, eval, dhp);
    }
    let root = tree.get(0).unwrap();
    debug!("Mcts root node: {}", root);
    let mcts_pri = mcts_tree_prior(&tree, 0, G::ACTN_SPC_SIZE);
    let mcts_post = mcts_tree_posterior(&tree, 0, G::ACTN_SPC_SIZE);
    let val_post = root.value / root.visits as f32;
    debug!("total probs should be 1.0. mcts_pri: {}  mcts_post: {}", total_prob(&mcts_pri), total_prob(&mcts_post));
    debug!("MCTS prior logits: {:?}", &mcts_pri);
    debug!("MCTS posterior logits:{:?}", &mcts_post);
    debug!("MCTS posterior value: {}", val_post);
    return (mcts_post, val_post);
}


// Performs a single MCTS simulation using PUCT
fn mcts_sim <G:GameState>
(
    tree: &mut MctsTree,
    state_orig: &G,
    eval: &mut impl Evaluator<G>,
    dhp: &DirichletHParams,
){
    let mut state = state_orig.clone();
    let mut node_id = 0; // Root
    loop{ // Drill down to a leaf node.
        let node = tree.get(node_id).unwrap();
        if node.is_expanded(){
            let (child_id, action) = tree.puct_select(node_id);
            node_id = child_id;
            state.apply_action(action);
        }else{
            break;
        }
    }
    // Is leaf node (possibly terminal)
    let value = 
        if state.is_terminal(){
            state.get_terminal_value() as f32
        }else{
            tree.expand(node_id, &state, eval, dhp)
        };
    tree.back_propogate(node_id, value); 
}


// Returns log mcts posterior probability distributions
// node_id must be a non-terminal and non-leaf node.
fn mcts_tree_posterior(tree: &MctsTree, node_id: NodeId, actn_spc_size: usize) -> Vec<f32>{
    let node = tree.get(node_id).unwrap();
    let mut p = vec![0.0; actn_spc_size];
    for child_id in tree.children(node_id){
        let child = tree.get(child_id).unwrap();
        let action = child.action.unwrap();
        p[action] = (child.visits as f32) / (node.visits as f32);
    }
    let mut p_sum: f32 = p.iter().sum();
    if p_sum == 0.0{
        // No child has been visited yet (only 1 simulation ran):
        // fall back to the (noised) priors.
        for child_id in tree.children(node_id){
            let child = tree.get(child_id).unwrap();
            p[child.action.unwrap()] = child.prior;
        }
        p_sum = p.iter().sum();
    }
    for e in p.iter_mut(){
        *e = (*e / p_sum).ln();
    }
    return p;
}

// Returns log probability
fn mcts_tree_prior(tree: &MctsTree, node_id: NodeId, actn_spc_size: usize) -> Vec<f32>{
    let mut p = vec![f32::NEG_INFINITY; actn_spc_size];
    for child_id in tree.children(node_id){
        let child = tree.get(child_id).unwrap();
        let action = child.action.unwrap();
        p[action] = child.prior.ln();
    }
    return p;
}


type NodeId = usize;

pub struct MctsNode {
    pub parent: Option<NodeId>,       // Parent index for backpropagation. None for root node.
    pub first_child: Option<NodeId>,  // First element in the children linked list.
                                       // None if node is terminal or not yet expanded.
    pub next_sibling: Option<NodeId>, // Next sibling index.
    pub action: Option<Action>,       // The action that led to this state
    pub prior: f32,                   // The prior probability of selecting this node from the parent.                 
    pub visits: u32,                  // Number of times this node has been visited. 1 + Sum(child_visits)
    pub value: f32,                   // Total value for this node.
                                      // If not expanded, is 0.0.
                                      // If expanded, value - sum(child_values).
    //pub is_terminal: bool,            // If this node is a terminal game state.            
}

impl MctsNode{
    pub fn is_expanded(&self) -> bool{
        return self.first_child != None; 
    }
}



impl fmt::Display for MctsNode{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut avg_val = 0.0;
        if self.visits > 0{
            avg_val = self.value / (self.visits as f32);
        }
        write!(f, "visits: {}   avg_value: {}   is_expanded: {}", self.visits, avg_val, self.is_expanded())?;
        return Ok(());
    }
}

pub struct Children<'a>{
    tree: &'a MctsTree,
    next_id: Option<NodeId>
}

impl <'a>Iterator for Children<'a>{
    type Item = NodeId;
    fn next(&mut self) -> Option<Self::Item> {
        let current_id = self.next_id?;
        self.next_id = self.tree.get(current_id)?.next_sibling;
        return Some(current_id);
    }
}


// Memory-pool implementation of MCTS Tree
// Root node has NodeId == 0;
pub struct MctsTree {
    nodes_: Vec<MctsNode>,
    c_puct: f32,
}

impl MctsTree{
    pub fn new(c_puct: f32) -> Self{
        let mut x = Self{
            nodes_: Vec::new(),
            c_puct: c_puct // c hyperparameter. Typically 1.0 to 2.5. More complex models will adjust this over time.
        };
        let root = MctsNode{
            parent: None,
            first_child: None,     
            next_sibling: None,
            action:  None,
            prior: 1.0,
            visits: 0,
            value: 0.0,
        };
        x.nodes_.push(root);
        return x;
    }
    
    pub fn get(&self, node_id: NodeId) -> Option<&MctsNode>{
        return self.nodes_.get(node_id);
    }

    pub fn get_mut(&mut self, node_id: NodeId) -> Option<&mut MctsNode>{
        return self.nodes_.get_mut(node_id);
    }


    fn children(&self, node_id: NodeId) -> Children<'_>{
        let first_child = self.get(node_id).unwrap().first_child;
        return Children { tree: &self, next_id: first_child }
    }
    
    fn back_propogate(&mut self, node_id: NodeId, value: f32){
        let mut curr_id = node_id;
        let mut polarity = 1.0;
        while let Some(curr) = self.get_mut(curr_id){
            curr.value += value * polarity;
            polarity *= -1.0;
            curr.visits += 1;
            if let Some(parent_id) = curr.parent{
                curr_id = parent_id;
            }else{
                break;
            }
        }
    }

    // Returns probabilistic upper confidence bound applied to trees (PUCT)
    // and returns best option.
    // Only call on node_id that have been expanded.
    // Panics if node_id has no children.
    fn puct_select(&self, node_id:NodeId) -> (NodeId, Action){
        let mut best_child_id = usize::MAX;
        let mut best_score = f32::NAN;
        for child_id in self.children(node_id){
            let score = self.calc_puct(child_id);
            if best_score.is_nan() || score > best_score{
                best_child_id = child_id;
                best_score = score;
            }
        }
        let action = self.get(best_child_id).unwrap().action.unwrap();
        return (best_child_id, action);
    }

    // Calculates PUCT score for a given node relative to its parent.
    // Node must have a parent. 
    fn calc_puct(&self, node_id: NodeId) -> f32{
        let node = self.get(node_id).unwrap();
        let parent = self.get(node.parent.unwrap()).unwrap();
        let exploitation_term =
            if node.visits == 0{
                0.0
            }else{
                -1.0 * node.value / (node.visits as f32)
            };
        let exploration_term = self.c_puct * node.prior *
                              (parent.visits as f32).sqrt() / ((1 + node.visits) as f32);
        return exploitation_term + exploration_term;
    }
    

    // Must not be called on a terminal state (debug-asserted; in release it
    // would evaluate a finished game and create children for it).
    // Returns the value of the node
    fn expand <G:GameState>(
        &mut self,
        node_id:NodeId,
        state: &G,
        eval: &mut impl Evaluator<G>,
        dhp: &DirichletHParams,
    ) -> f32{
        debug_assert_eq!(state.is_terminal(), false);
        trace!("Expanding node {}", node_id);
        let (mut p, v) = eval.evaluate(state);
        let legal_mask = state.legal_actions_mask();
        if node_id == 0{
            trace!("Adding dirichlet noise {} to root", dhp);
            p = add_dirichlet_noise(p, &legal_mask, dhp);
        }
        for action in 0..p.len(){
            // Every legal action gets a child, even if its prior underflowed to 0.
            if legal_mask[action] != 0.0{
                self.add_child(action, node_id, p[action]);
            }
        }
        return v;
    }

    // Create a new node in the arena-pool. Return it's id.
    fn add_child(&mut self, action: Action, parent: NodeId, prior:f32) -> NodeId{
        let child_id: NodeId = self.nodes_.len();
        let child = MctsNode{
            action: Some(action),
            parent: Some(parent),
            first_child: None,
            next_sibling: self.nodes_[parent].first_child,
            prior: prior,
            visits: 0,
            value: 0.0,
        };
        self.nodes_.push(child);
        self.nodes_[parent].first_child = Some(child_id);
        return child_id;
    }

}


// p is (potentailly unormalized) log probabilities.
// Returns the sum(exp(p));
fn total_prob(p: &Vec<f32>)->f32{
    let mut sum = 0.0;
    for f in p{
        sum += f.exp();
    }
    return sum;
}

// exponentiate ever element in the vector
fn exp (v: &Vec<f32>) -> Vec<f32>{
    let mut o = Vec::with_capacity(v.len());
    for e in v.iter(){
        o.push((*e).exp());
    }
    return o;
}





#[derive(Debug, Clone)]
pub struct DirichletHParams{
    pub epsilon: f32, // [0,1]. Mixing param. What percent of prior is from dirichlet noise
                  // Set to 0 to disable dirichlet noise.
    pub alpha: f32, // Concentration parameter.
}

impl DirichletHParams{
    pub fn from_gamestate<G:GameState>() -> Self{
        Self {
            epsilon: 0.25,
            alpha: 10.0 / G::ACTN_SPC_SIZE as f32,
        }
    }
}

impl Default for DirichletHParams{
    fn default() -> Self {
        Self {
            epsilon: 0.0,
            alpha: 1.0,
        }
    }
}


impl Display for DirichletHParams{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DirichletHParams{{epislon: {}, alpha: {} }}",
               self.epsilon, self.alpha).unwrap();
        return Ok(());
    }
}

// Mixes Dirichlet noise into p over the legal actions only. p must already be
// a probability distribution over legal actions, so the result still sums to 1.
// No-op if noise is disabled or there are fewer than 2 legal actions
// (Dirichlet needs at least 2 components).
fn add_dirichlet_noise(mut p: Vec<f32>, legal_mask: &[f32], dhp: &DirichletHParams) -> Vec<f32>{
    if dhp.epsilon == 0.0{
        return p;
    }
    let legal: Vec<usize> = (0..p.len()).filter(|&i| legal_mask[i] != 0.0).collect();
    if legal.len() < 2{
        return p;
    }
    let alpha_vec = vec![dhp.alpha; legal.len()];
    let dirichlet = Dirichlet::new(&alpha_vec).unwrap();
    let noise = dirichlet.sample(&mut rand::rng());
    for (&i, &n) in legal.iter().zip(noise.iter()){
        p[i] = (1.0 - dhp.epsilon) * p[i] + dhp.epsilon * n;
    }
    return p;
}