//! Structs describing a single worker thread

use std::{cell::UnsafeCell, marker::PhantomData, sync::{self, Mutex}, thread::{JoinHandle, spawn}};
use std::sync::Arc;

use sync::Condvar;

use super::{CodeExecutor, CodeGenerator, ThreadContext};

pub(crate) enum CompilationStatus {
    /// Compilation unfinished
    Pending,
    /// Compilation finished successfully
    Done,
    /// Compilation failed
    Error(String)
}

impl CompilationStatus {
    /// Returns `true` if the compilation_status is [`Pending`].
    pub(crate) fn is_pending(&self) -> bool {
        matches!(self, Self::Pending)
    }
}

/// Enum designating the *next* task of the worker thread
#[derive(Clone, Debug)]
pub(crate) enum WorkerTask {
    /// Remain idle
    Idle,
    /// Run for a given number of steps
    Run(usize),
    /// Stop worker thread
    Stop
}

impl WorkerTask {
    /// Returns `true` if the worker_task is [`Idle`].
    pub(crate) fn is_idle(&self) -> bool {
        matches!(self, Self::Idle)
    }
}

/// Enum designating the *current* status of the worker
pub(crate) enum WorkerStatus {
    Busy,
    Idle,
    Dead
}

impl WorkerStatus {
    /// Returns `true` if the worker_status is [`Busy`].
    pub(crate) fn is_busy(&self) -> bool {
        matches!(self, Self::Busy)
    }
}

/// Representation of a single worker thread
pub(crate) struct WorkerThread {
    thread: JoinHandle<()>,
    task_flag: Arc<(Mutex<WorkerTask>, Condvar)>,
    compilation_flag: Arc<(Mutex<CompilationStatus>, Condvar)>,
    status_flag: Arc<(Mutex<WorkerStatus>, Condvar)>,
    // Force !Sync on this structure: Concurrent access from multiple threads on this
    // can lead to data races (worker running while another thread accesses data in
    // the global state)
    // This is a fairly sloppy bandaid for the fundamental problem, that we do not
    // tell Rust anywhere about the data that is accessed mutably by the generated code
    // => TODO: Do better
    _no_sync: PhantomData<UnsafeCell<()>> // This blocks read access from more than one thread
}

impl WorkerThread {
    pub(crate) fn spawn(context: ThreadContext) -> Self {
        let task_flag = Arc::new((Mutex::new(WorkerTask::Idle), Condvar::new()));
        let task_flag_clone = task_flag.clone();
        let compilation_flag = Arc::new((Mutex::new(CompilationStatus::Pending), Condvar::new()));
        let compilation_flag_clone = compilation_flag.clone();
        let status_flag = Arc::new((Mutex::new(WorkerStatus::Idle), Condvar::new()));
        let status_flag_clone = status_flag.clone();
        let thread = spawn(move || {
            // Move stuff to thread
            let context = context;
            let task_flag = task_flag_clone;
            let compilation_flag = compilation_flag_clone;
            let status_flag = status_flag_clone;

            // Compile code
            let compilation_result = (|| {
                let codegen = CodeGenerator::new(context)?;
                CodeExecutor::new(codegen)
            })();
            // Set compilation status appropriately
            let set_compilation_status = |status| {
                let (compilation_lock, compilation_cvar) = &*compilation_flag;
                let mut compilation_status = compilation_lock.lock().unwrap();
                *compilation_status = status;
                compilation_cvar.notify_all();
            };
            let mut executor = match compilation_result {
                Ok(executor) => {
                    set_compilation_status(CompilationStatus::Done);
                    executor
                },
                Err(error) => {
                    set_compilation_status(CompilationStatus::Error(error.to_string()));
                    return;
                }
            };
            // Worker main loop
            loop {
                // Wait while task is idle
                let (task_lock, task_cvar) = &*task_flag;
                let mut task = task_lock.lock().unwrap();
                task = task_cvar.wait_while(task, |task| task.is_idle()).unwrap();
                // Set status to busy
                let (status_lock, status_cvar) = &*status_flag;
                let mut status = status_lock.lock().unwrap();
                *status = WorkerStatus::Busy;
                std::mem::drop(status);
                status_cvar.notify_all();
                // Worker has taken task => set next task to idle and unlock task lock
                let tmp = task.clone();
                *task = WorkerTask::Idle;
                std::mem::drop(task);
                task_cvar.notify_all();
                let task = tmp;
                match task {
                    WorkerTask::Idle => unreachable!(),
                    WorkerTask::Run(num) => {
                        for _ in 0..num {
                            executor.run()
                        }
                    }
                    WorkerTask::Stop => { break; }
                };
                // Check if we have gotten a new task in the meantime
                let task = task_lock.lock().unwrap();
                if task.is_idle() {
                    // Set status to idle
                    let (status_lock, status_cvar) = &*status_flag;
                    let mut status = status_lock.lock().unwrap();
                    *status = WorkerStatus::Idle;
                    std::mem::drop(status);
                    status_cvar.notify_all();
                }
                std::mem::drop(task);

                // TODO: This is actually unsound: Thread A might wait while
                // thread B creates a new task => the worker starts running while
                // thread A still assumes idle status (i.e. has access to the data store)
                // Can we solve this by simply making WorkerThread !Sync?
                // Or maybe make some more complex task queue struct that blocks once one
                // thread is waiting? But how do we synchronize between workers then?
            };
            // Set status to dead
            let (status_lock, status_cvar) = &*status_flag;
            let mut status = status_lock.lock().unwrap();
            *status = WorkerStatus::Dead;
            std::mem::drop(status);
            status_cvar.notify_all();
        });
        Self {
            thread,
            task_flag,
            compilation_flag,
            status_flag,
            _no_sync: PhantomData
        }
    }

/* TODO: Rework the synchronization here */

    pub(crate) fn wait_for_compilation(&self) {
        let (compilation_lock, compilation_cvar) = &*self.compilation_flag;
        let mut compilation_status = compilation_lock.lock().unwrap();
        compilation_status = compilation_cvar.wait_while(compilation_status,
            |compilation_status| compilation_status.is_pending()).unwrap();
        // TODO: Check for correct result
        match &*compilation_status {
            CompilationStatus::Error(e) => panic!("{}", e),
            CompilationStatus::Done => {}
            CompilationStatus::Pending => unreachable!(),
        }
    }

    pub(crate) fn run_step(&self) {
        let (task_lock, task_cvar) = &*self.task_flag;
        let mut task = task_lock.lock().unwrap();
        *task = match *task {
            WorkerTask::Idle => WorkerTask::Run(1),
            WorkerTask::Run(num) => WorkerTask::Run(num+1),
            WorkerTask::Stop => WorkerTask::Stop
        }; 
        task_cvar.notify_all();
    }

    pub(crate) fn wait(&self) {
        // Wait for task queue to be empty
        let (task_lock, task_cvar) = &*self.task_flag;
        let mut task = task_lock.lock().unwrap();
        task = task_cvar.wait_while(task, |task| !task.is_idle()).unwrap();
        std::mem::drop(task);
        // Wait for last task to finish
        let (status_lock, status_cvar) = &*self.status_flag;
        let mut status = status_lock.lock().unwrap();
        status = status_cvar.wait_while(status, |status| status.is_busy()).unwrap();
        std::mem::drop(status);
    }

    pub(crate) fn join(self) {
        let (task_lock, task_cvar) = &*self.task_flag;
        let mut task = task_lock.lock().unwrap();
        *task = WorkerTask::Stop;
        std::mem::drop(task);
        task_cvar.notify_all();
        self.thread.join().expect("Cannot join threads");
    }
}