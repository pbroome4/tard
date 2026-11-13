//////////////////////////////////////////////////////////////////////////////
//
//                      Batched (Centralized) Evaluator
//
//////////////////////////////////////////////////////////////////////////////
// Clients send evaluation requests to a single server thread, which batches
// all pending requests into one forward pass. Model updates are sent to the
// server thread, so every client always uses the server's current model.

use std::collections::VecDeque;
use std::marker::PhantomData;
use std::thread::{self, JoinHandle};
use burn::tensor::{Device, Tensor, TensorData};
use burn::module::Module;
use crossbeam::channel::{Receiver, RecvError, Sender, TryRecvError};
use tracing::{trace, debug, info, warn};

use crate::games::*;
use crate::model::Tard;
use crate::util::panic_message;
use super::legalize_policy;
use std::time::Instant;
use super::{Evaluator, EvalBackend, EvalStats, EvalStatsSnapshot};


// The client encodes the state and legalizes the returned policy itself, so
// that per-position CPU work runs in parallel on the worker threads rather
// than serially on the server thread.
#[derive(Debug, Clone)]
pub struct BatchedClient{
    s_eval: Sender<EvalRequest>
}

impl <G:GameState> Evaluator<G> for BatchedClient{
    fn evaluate(&mut self, g: &G) -> (Vec<f32>, f32){
        // A fresh one-shot channel per request (rather than one reused per
        // client) means the only sender lives in the request: if the server
        // panics, the request is dropped and recv() fails instead of
        // blocking forever.
        let (s,r) = crossbeam::channel::bounded(1);
        let req = EvalRequest{
            input: g.to_tensor(),
            resp_chan: s
        };
        self.s_eval.send(req)
            .expect("Batched eval server is gone (it likely panicked; see its error)");
        let resp = r.recv()
            .expect("Batched eval server dropped the request (it likely panicked; see its error)");
        return (legalize_policy(resp.p, g), resp.v);
    }
}


pub struct BatchedBackend<G>{
    s_update: Sender<Tard>,
    s_eval:  Sender<EvalRequest>,
    server: Option<JoinHandle<()>>,
    stats: EvalStats, // recorded by the server thread
    _marker: PhantomData<fn() -> G>,
}

impl <G:GameState> BatchedBackend<G>{
    pub fn new(t: &Tard, max_batch_size: usize) -> Self{
        let (s_u, r_u) = crossbeam::channel::unbounded();
        let (s_e, r_e) = crossbeam::channel::unbounded();
        let stats = EvalStats::default();
        let server_stats = stats.clone();
        let server = thread::Builder::new()
            .name("batched-eval-server".to_string())
            .spawn(move ||{
                run_server::<G>(r_u, r_e, max_batch_size, server_stats)
            })
            .expect("Failed to spawn batched eval server thread");
        let mut slf = Self {
            s_update: s_u,
            s_eval: s_e,
            server: Some(server),
            stats,
            _marker: PhantomData,
        };
        EvalBackend::<G>::update(&mut slf, t);
        return slf;
    }
}

impl <G:GameState> EvalBackend<G> for BatchedBackend<G>{
    type Client = BatchedClient;

    fn client(&mut self) -> BatchedClient{
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

    fn stats(&self) -> EvalStatsSnapshot{
        return self.stats.snapshot();
    }
}


// Server loop. Exits once either the update channel (backend dropped) or
// the request channel (backend and all clients dropped) disconnects.
fn run_server<G:GameState>(
    r_u: Receiver<Tard>,
    r_e: Receiver<EvalRequest>,
    max_batch_size: usize,
    stats: EvalStats)
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
    let mut q = VecDeque::<EvalRequest>::new();
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
        if !q.is_empty(){
            let batch_size = std::cmp::min(q.len(), max_batch_size);
            trace!("Computing batch of size {}", batch_size);
            let start = Instant::now();
            let b_tens = batch_eval_tensor::<G>(&q, &device, batch_size);
            let (p_batch, v_batch) = t.forward(b_tens);
            let p_batch_vec = p_batch.exp()
                                     .into_data()
                                     .try_into_vec::<f32>()
                                     .unwrap();
            let v_batch_vec = v_batch.into_data()
                                     .try_into_vec::<f32>()
                                     .unwrap();
            stats.record(batch_size, start.elapsed());
            let responses = p_batch_vec.chunks_exact(G::ACTN_SPC_SIZE).zip(v_batch_vec);
            for (req, (p, v)) in q.drain(0..batch_size).zip(responses){
                trace!("Batch element has p: {:?}, v: {}", p, v);
                if let Err(x) = req.resp_chan.send(EvalResponse{p: p.to_vec(), v}){
                    warn!("Batched eval response failed to send: {}", x);
                }
            }
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

// Stacks the first batch_size requests' encoded states into one input tensor.
fn batch_eval_tensor <G:GameState>(
    q: &VecDeque<EvalRequest>,
    device: &Device,
    batch_size: usize
) -> Tensor<4>{
    let mut v = Vec::with_capacity(batch_size * G::N_CHAN * G::ROWS * G::COLS);
    for req in q.iter().take(batch_size){
        v.extend_from_slice(&req.input);
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
fn read_queue(
    r: &Receiver<EvalRequest>,
    q: &mut VecDeque<EvalRequest>) -> (usize, bool)
{
    let n = q.len();
    let mut is_connected = true;
    // Empty queue can just block read because no other requests are waiting
    if q.is_empty(){
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
struct EvalRequest{
    input: Vec<f32>,                   // GameState::to_tensor() of the position
    resp_chan: Sender<EvalResponse>    // one-shot response channel
}
