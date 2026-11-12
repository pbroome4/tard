//////////////////////////////////////////////////////////////////////////////
//
//                      Batched (Centralized) Evaluator
//
//////////////////////////////////////////////////////////////////////////////
// Clients send evaluation requests to a single server thread, which batches
// all pending requests into one forward pass. Model updates are sent to the
// server thread, so every client always uses the server's current model.

use std::collections::VecDeque;
use std::thread::{self, JoinHandle};
use burn::tensor::{Device, Tensor, TensorData};
use burn::module::Module;
use crossbeam::channel::{Receiver, RecvError, Sender, TryRecvError};
use tracing::{trace, debug, info, warn};

use crate::gamestate::*;
use crate::tard::*;
use crate::utils::*;
use super::{Evaluator, EvalBackend};


#[derive(Debug, Clone)]
pub struct BatchedClient<G>{
    s_eval: Sender<EvalRequest<G>>
}

impl <G:GameState> Evaluator<G> for BatchedClient<G>{
    fn evaluate(&mut self, g: &G) -> (Vec<f32>, f32){
        let (s,r) = crossbeam::channel::bounded(1);
        let req = EvalRequest{
            g: g.clone(),
            resp_chan: s
        };
        self.s_eval.send(req)
            .expect("Batched eval server is gone (it likely panicked; see its error)");
        let resp = r.recv()
            .expect("Batched eval server dropped the request (it likely panicked; see its error)");
        return (resp.p, resp.v);
    }
}


pub struct BatchedBackend<G>{
    s_update: Sender<Tard>,
    s_eval:  Sender<EvalRequest<G>>,
    server: Option<JoinHandle<()>>,
}

impl <G:GameState> BatchedBackend<G>{
    pub fn new(t: &Tard, max_batch_size: usize) -> Self{
        let (s_u, r_u) = crossbeam::channel::unbounded();
        let (s_e, r_e) = crossbeam::channel::unbounded();
        let server = thread::Builder::new()
            .name("batched-eval-server".to_string())
            .spawn(move ||{
                run_server::<G>(r_u, r_e, max_batch_size)
            })
            .expect("Failed to spawn batched eval server thread");
        let mut slf = Self {
            s_update: s_u,
            s_eval: s_e,
            server: Some(server),
        };
        EvalBackend::<G>::update(&mut slf, t);
        return slf;
    }
}

impl <G:GameState> EvalBackend<G> for BatchedBackend<G>{
    type Client = BatchedClient<G>;

    fn client(&mut self) -> BatchedClient<G>{
        return BatchedClient{s_eval: self.s_eval.clone()};
    }

    fn update(&mut self, t: &Tard){
        info!("Sending batched eval server updated tard");
        if self.s_update.send(t.valid()).is_err(){
            // The server only drops its receiver by exiting; surface why.
            let reason = match self.server.take().map(|h| h.join()){
                Some(Err(payload)) => format!("panicked: {}", panic_message(&payload)),
                Some(Ok(())) => "exited".to_string(),
                None => "already reported".to_string(),
            };
            panic!("Batched eval server is gone ({})", reason);
        }
    }
}


// Server loop. Exits once either the update channel (backend dropped) or
// the request channel (backend and all clients dropped) disconnects.
fn run_server<G:GameState>(
    r_u: Receiver<Tard>,
    r_e: Receiver<EvalRequest<G>>,
    max_batch_size: usize)
{
    info!("Batched eval server starting, awaiting Tard");
    let Ok(mut t) = r_u.recv() else {
        return;
    };
    info!("Batched eval server received Tard");
    let device = t.devices()
                  .into_iter()
                  .next()
                  .unwrap();
    let mut q = VecDeque::<EvalRequest<G>>::new();
    let mut is_running = true;
    while is_running{
        trace!("Checking for Tard updates");
        let (t_updated, recv_connected) = check_for_model_update(&r_u, t);
        t = t_updated;
        is_running &= recv_connected;

        if q.len() < 10 * max_batch_size{
            let (n, recv_connected) = read_queue(&r_e, &mut q);
            trace!("Added {} elements to queue", n);
            is_running &= recv_connected;
        }
        if q.len() > 0{
            let batch_size = std::cmp::min(q.len(), max_batch_size);
            trace!("Computing batch of size {}", batch_size);
            let b_tens = batch_eval_tensor(&q, &device, batch_size);
            let (p_batch, v_batch) = t.forward(b_tens);
            let mut p_batch_vec = p_batch.exp()
                                         .into_data()
                                         .try_into_vec::<f32>()
                                         .unwrap();
            let mut v_batch_vec = v_batch.into_data()
                                         .try_into_vec::<f32>()
                                         .unwrap();
            // count backward because efficient to delete from end of vector
            for i in (0..batch_size).rev(){
                let p = p_batch_vec.split_off(p_batch_vec.len() - G::ACTN_SPC_SIZE);
                let p = legalize_policy(p, &q[i].g);
                let v = v_batch_vec.pop().unwrap();
                trace!("Batch element {}/{} has p: {:?}, v: {}",
                        i+1, batch_size, &p, &v);
                if let Err(x) = q[i].resp_chan.send(EvalResponse{p, v}){
                    warn!("Batched eval response failed to send: {}", x);
                }
            }
            q.drain(0..batch_size);
        }
    }
    debug!("Batched eval server shutting down");
}

// Drains all pending model updates, keeping the latest.
// Returns (latest model, is_connected).
fn check_for_model_update(r_tard: &Receiver<Tard>, t_in: Tard) -> (Tard, bool){
    let mut t = t_in;
    loop{
        match r_tard.try_recv(){
            Ok(t_updated) => {
                info!("Batched eval server updating model");
                t = t_updated;
            }
            Err(TryRecvError::Empty) => {
                return (t, true);
            }
            Err(TryRecvError::Disconnected) => {
                info!("Tard update channel closed... signaling batched eval server shutdown");
                return (t, false);
            }
        }
    }
}

fn batch_eval_tensor <G:GameState>(
    q: &VecDeque<EvalRequest<G>>,
    device: &Device,
    batch_size: usize
) -> Tensor<4>{
    let mut v = Vec::with_capacity(batch_size * G::N_CHAN * G::ROWS * G::COLS);
    for i in 0..batch_size{
        v.extend_from_slice(&q[i].g.to_tensor());
    }
    let td = TensorData::new(v, [batch_size, G::N_CHAN, G::ROWS, G::COLS]);
    return Tensor::<4>::from_data(td, device);
}


// Receives all elements currently in the channel and adds them to queue
// then returns. If the queue is currently empty, it will block until
// at least 1 element is read. Otherwise non-blocking
//
// Returns
//      Number of elements read into queue.
//      is_connected
fn read_queue<G:GameState>(
    r: &Receiver<EvalRequest<G>>,
    q: &mut VecDeque<EvalRequest<G>>) -> (usize, bool)
{
    let n = q.len();
    let mut is_connected = true;
    // Empty queue can just block read because no other requests are waiting
    if q.len() == 0{
        trace!("Queue is empty, performing blocking read");
        match r.recv(){
            Ok(req) => q.push_back(req),
            Err(RecvError) => return (0, false), // disconnected
        }
    }
    loop{
        match r.try_recv(){
            Ok(req) => q.push_back(req),
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => {
                is_connected = false;
                break;
            }
        }
    }
    return (q.len() - n, is_connected);
}


#[derive(Debug)]
struct EvalResponse{
    p: Vec<f32>,
    v: f32
}

#[derive(Debug)]
struct EvalRequest<G>{
    g: G,
    resp_chan: Sender<EvalResponse>    // one-shot response channel
}
