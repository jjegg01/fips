//! Structures related to the callback mechanism (i.e. return to Rust from a
//! running FIPS simulation)

use super::{CallbackMessage, ThreadContext, analysis::{BarrierID, BarrierKind}};

pub struct CallbackTarget {
    /// Thread context (contains all the synchronization primitives we need)
    pub(crate) thread_context: ThreadContext,
}

impl CallbackTarget {
    pub(crate) fn new(thread_context: ThreadContext) -> Self {
        Self {
            thread_context
        }
    }

    pub(crate) fn handle_call2rust(&self, barrier_id: BarrierID) {
        assert!(matches!(
            self.thread_context.executor_context.global_context.simgraph.barriers.get(barrier_id).unwrap().kind,
            BarrierKind::CallBarrier(_)
        ));
        self.thread_context.executor_context.call_sender.send(CallbackMessage::Call(barrier_id))
            .expect("Callback thread has hung up");
        self.thread_context.executor_context.call_end_barrier.wait();
    }

    pub(crate) fn handle_interaction(&self, barrier_id: BarrierID, block_index: usize) -> (*const usize, *const usize) {
        match self.thread_context.executor_context.global_context.simgraph.barriers.get(barrier_id).unwrap().kind {
            BarrierKind::InteractionBarrier(interaction_id, _) => {
                let barrier = self.thread_context.executor_context.barriers.get(&barrier_id).unwrap();
                let result = barrier.wait();
                if result.is_leader() {
                    let step = *self.thread_context.executor_context.step_counter.read().unwrap();
                    let particle_index = &self.thread_context.executor_context.global_context.runtime.particle_index;
                    let particle_store = &self.thread_context.executor_context.global_context.runtime.particle_store;
                    self.thread_context.executor_context.neighbor_lists.get(&interaction_id).unwrap()
                        .write().unwrap().rebuild_if_required(step, particle_index, particle_store);
                    barrier.wait();
                }
                else {
                    barrier.wait();
                }
                let neighbor_list = self.thread_context.executor_context.neighbor_lists.get(&interaction_id)
                    .unwrap().read().unwrap();
                let (neighbor_list_index, neighbor_list) = &neighbor_list.neighbor_lists[block_index];
                (neighbor_list_index.as_ptr(), neighbor_list.as_ptr())
            }
            _ => panic!()
        }
    }

    pub(crate) fn handle_interaction_sync(&self, barrier_id: BarrierID) {
        match self.thread_context.executor_context.global_context.simgraph.barriers.get(barrier_id).unwrap().kind {
            BarrierKind::InteractionBarrier(_, _) => {
                let barrier = self.thread_context.executor_context.barriers.get(&barrier_id).unwrap();
                barrier.wait();
            }
            _ => panic!()
        }
    }

    pub(crate) fn end_of_step(&self) {
        // Make all workers synchronize at the end of each step
        let result = self.thread_context.executor_context.step_barrier.wait();
        if result.is_leader() {
            *self.thread_context.executor_context.step_counter.write().unwrap() += 1;
        }
        self.thread_context.executor_context.step_barrier.wait();
    }
}