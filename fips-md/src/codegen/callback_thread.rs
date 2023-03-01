//! Thread for executing "call to Rust" callbacks

use std::{any::Any, collections::{BTreeSet, HashMap}, sync::{self, Arc, mpsc}, thread::{JoinHandle, spawn}};

use anyhow::Result;

use super::{GlobalContext, analysis::BarrierID};

pub type CallbackType = fn(&Arc<GlobalContext>, &mut Box<dyn Any + Send>);
pub type CallbackStateType = Box<dyn Any + Send>;

pub(crate) enum CallbackMessage {
    Call(BarrierID),
    Register(BTreeSet<BarrierID>, CallbackType, CallbackStateType),
    Unregister(BTreeSet<BarrierID>),
    Quit
}

/// Registry for callbacks: Every callback can be associated with one ore more barriers
struct CallbackRegistry {
    registry: HashMap<BTreeSet<BarrierID>, (CallbackType, CallbackStateType)>
}

impl CallbackRegistry {
    fn new() -> Self {
        Self {
            registry: HashMap::new()
        }
    }

    fn insert(&mut self, barriers: BTreeSet<BarrierID>,
        callback: CallbackType, state: CallbackStateType)
    {
        // TODO: Assert disjointness of barriers in registry?
        self.registry.insert(barriers, (callback, state));
    }

    // fn get(&self, barrier: BarrierID) -> Option<&(fn(GlobalContext, &mut Box<dyn Any + Send>), Box<dyn Any>)> {
    //     for (barriers, callback_data) in self.registry.iter() {
    //         if barriers.contains(&barrier) {
    //             return Some(callback_data)
    //         }
    //     }
    //     None
    // }

    fn get_mut(&mut self, barrier: BarrierID) -> Option<&mut (CallbackType, CallbackStateType)> {
        for (barriers, callback_data) in self.registry.iter_mut() {
            if barriers.contains(&barrier) {
                return Some(callback_data)
            }
        }
        None
    }

    fn remove(&mut self, barriers: &BTreeSet<BarrierID>) -> Option<(CallbackType, CallbackStateType)> {
        self.registry.remove(barriers)
    }
}

pub(crate) struct CallbackThread {
    /// Sender for tasks to the callback thread
    sender: mpsc::Sender<CallbackMessage>,
    /// Receiver for getting callback data out of the callback thread
    unregister_receiver: mpsc::Receiver<(CallbackType, CallbackStateType)>,
    /// The actual handle to the callback thread
    thread: JoinHandle<()>
}

impl CallbackThread {
    /// Create a new callback thread that will wait on the given barrier after each call
    pub fn new(call_end_barrier: Arc<sync::Barrier>, num_workers: usize,
        global_context: Arc<GlobalContext>) 
    -> Self {
        // Create MPSC channel for communication between workers and callback thread
        let (sender, receiver) = mpsc::channel();
        let (unregister_sender, unregister_receiver) = mpsc::channel();
        // Spawn callback thread
        let thread = spawn(move || {
            // Callback registry
            let mut callbacks = CallbackRegistry::new();
            
            loop {
                // Wait until every worker has ordered us to do the callback
                let mut callback_barrier = None;
                for i in 0..num_workers {
                    let message = receiver.recv().expect("Callback channel senders have hung up");
                    match message {
                        CallbackMessage::Quit => { return },
                        CallbackMessage::Register(barriers, callback, state) => {
                            if i != 0 {
                                panic!("Got callback registration while waiting for workers")
                            }
                            callbacks.insert(barriers, callback, state);
                            break;
                        },
                        CallbackMessage::Unregister(barriers) => {
                            if i != 0 {
                                panic!("Got callback unregistration while waiting for workers")
                            }
                            // TODO: Fail more gracefully?
                            let callback_data = callbacks.remove(&barriers)
                                .expect("Cannot unregister barrier set: No callback defined.");
                            unregister_sender.send(callback_data)
                                .expect("Callback return channel has hung up.");
                            break;
                        },
                        CallbackMessage::Call(barrier) => {
                            match &mut callback_barrier {
                                None => {
                                    callback_barrier = Some(barrier);
                                    continue;
                                }
                                Some(previous_barrier) => {
                                    // Verify that we got the right call barrier
                                    if *previous_barrier == barrier {
                                        continue;
                                    }
                                    // Panic otherwise
                                    else {
                                        panic!("Got conflicting callback barrier IDs!");
                                    }
                            }}
                    }}
                }
                if let Some(callback_barrier) = callback_barrier {
                    // Now callback_barrier is consistent and we can call the correct function
                    if let Some((callback, state)) = callbacks.get_mut(callback_barrier) {
                        callback(&global_context, state);
                    }
                    // Finally unlock the call_end barrier
                    call_end_barrier.wait();
                }
            }
        });
        Self {
            sender,
            unregister_receiver,
            thread
        }
    }

    /// Get a new sender for communication with the callback thread
    pub(crate) fn get_sender(&self) -> mpsc::Sender<CallbackMessage> {
        self.sender.clone()
    }

    pub(crate) fn join(self) {
        self.sender.send(CallbackMessage::Quit).expect("Callback thread has hung up");
        self.thread.join().expect("Callback thread has panicked at some point");
    }

    pub fn register_callback(&mut self, barriers: BTreeSet<BarrierID>,
        callback: CallbackType,
        callback_state: CallbackStateType) 
    -> Result<()> {
        // TODO: Check if callback already existed
        self.sender.send(CallbackMessage::Register(barriers, callback, callback_state))
            .expect("Callback thread has hung up");
        Ok(())
    }

    pub fn unregister_callback(&mut self, barriers: BTreeSet<BarrierID>) 
    -> Result<(CallbackType, CallbackStateType)> {
        self.sender.send(CallbackMessage::Unregister(barriers))
            .expect("Callback thread has hung up");
        let callback_data = self.unregister_receiver.recv()
            .expect("Callback thread has hung up");
        Ok(callback_data)
    }
}