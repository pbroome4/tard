//! Training examples and the bounded replay buffer they are sampled from.

use bounded_vec_deque::BoundedVecDeque;
use rand::RngExt;

use crate::games::GameState;
use crate::selfplay::SelfPlayElement;

////// Training structures and Batch Utilities ////
#[derive(Clone, Debug)]
pub struct TrainingElement {
    pub state: Vec<f32>, // flattened N_CHAN x ROWS x COLS (GameState::to_tensor)
    pub mcts_posterior: Vec<f32>, // MCTS visit distribution, length ACTN_SPC_SIZE
    pub outcome: f32,    // game result from the side-to-move's perspective
}

pub fn selfplay_to_training<G: GameState>(sp_vec: Vec<SelfPlayElement<G>>) -> Vec<TrainingElement> {
    let mut te_vec = Vec::with_capacity(sp_vec.len());
    for spe in sp_vec.into_iter() {
        let te = TrainingElement {
            state: spe.g.to_tensor(),
            mcts_posterior: spe.mcts_post,
            outcome: spe.outcome,
        };
        te_vec.push(te);
    }
    return te_vec;
}

pub struct TrainingBuffer {
    memory_: BoundedVecDeque<TrainingElement>,
}

impl TrainingBuffer {
    pub fn new(cap: usize) -> Self {
        return Self {
            memory_: BoundedVecDeque::with_capacity(cap, cap),
        };
    }

    // Samples `size` elements uniformly with replacement.
    // Returns an empty Vec if the buffer is empty.
    pub fn sample(&self, size: usize) -> Vec<&TrainingElement> {
        let mut v = Vec::with_capacity(size);
        let l = self.memory_.len();
        if l > 0 {
            let mut rng = rand::rng();
            for _ in 0..size {
                v.push(&self.memory_[rng.random_range(0..l)]);
            }
        }
        return v;
    }

    // Appends elements at the back, evicting the oldest from the front
    // once the buffer is at capacity.
    pub fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = TrainingElement>,
    {
        return self.memory_.extend(iter);
    }

    pub fn len(&self) -> usize {
        return self.memory_.len();
    }

    pub fn is_empty(&self) -> bool {
        return self.memory_.is_empty();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::minipente::MiniPenteState;

    fn element(outcome: f32) -> TrainingElement {
        return TrainingElement {
            state: vec![],
            mcts_posterior: vec![],
            outcome,
        };
    }

    #[test]
    fn training_buffer_evicts_the_oldest_elements_at_capacity() {
        let mut buf = TrainingBuffer::new(3);
        assert!(buf.is_empty());
        buf.extend((0..5).map(|i| element(i as f32)));
        assert_eq!(buf.len(), 3);
        let kept: Vec<f32> = buf.memory_.iter().map(|e| e.outcome).collect();
        assert_eq!(kept, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn training_buffer_samples_with_replacement() {
        let mut buf = TrainingBuffer::new(10);
        assert!(
            buf.sample(5).is_empty(),
            "sampling an empty buffer returns nothing"
        );
        buf.extend((0..3).map(|i| element(i as f32)));
        let s = buf.sample(50);
        assert_eq!(s.len(), 50);
        assert!(s.iter().all(|e| (0.0..3.0).contains(&e.outcome)));
    }

    #[test]
    fn selfplay_to_training_keeps_targets_and_encodes_state() {
        let g = MiniPenteState::new();
        let spe = SelfPlayElement {
            g: g.clone(),
            mcts_post: vec![0.5, 0.5],
            outcome: -1.0,
        };
        let te = selfplay_to_training(vec![spe]);
        assert_eq!(te.len(), 1);
        assert_eq!(te[0].state, g.to_tensor());
        assert_eq!(te[0].mcts_posterior, vec![0.5, 0.5]);
        assert_eq!(te[0].outcome, -1.0);
    }
}
