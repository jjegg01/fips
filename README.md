[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7689625.svg)](https://doi.org/10.5281/zenodo.7689625)
[![Funding provided by DFG Project ID 433682494 - SFB 1459](https://img.shields.io/badge/DFG%20funded-Project%20ID%20433682494%20--%20SFB%201459%20-blue)](https://gepris.dfg.de/gepris/projekt/433682494?context=projekt&task=showDetail&id=433682494&)
[![crates.io](https://img.shields.io/crates/v/fips_md.svg)](https://crates.io/crates/fips_md)
[![project chat](https://img.shields.io/badge/zulip-join_chat-brightgreen.svg)](https://fips-md.zulipchat.com)

FIPS - The Fearlessly Integrating Particle Simulator
====================================================

About
-----

FIPS is a framework for simulating arbitrary particle dynamics written in and inspired by the Rust programming language. Unlike most other MD frameworks it does not make any assumptions on what kind of particle types, integration schemes or particle interactions you want. Instead, you can freely define any kind of numerical schemes you need in a domain-specific language and have FIPS do all the heavy-lifting for you. FIPS is specifically designed to run well on shared-memory systems while keeping the woes of parallel programming away from you. You can read more about the theory behind FIPS in [this article](https://arxiv.org/abs/2302.14170).

Prerequisites
-------------
In the following we assume that you are using a Linux system. Building FIPS for other platforms is not tested at the moment.

The only non-Rust dependency of FIPS is the LLVM Compiler framework. Here are the steps to installing it from source:

1. Install the required software for building LLVM. You can find a detailed list [here](https://llvm.org/docs/GettingStarted.html#requirements).
2. Download the latest version of LLVM 14 from the [website of the LLVM project](https://releases.llvm.org/). **It is very important that you use version 14! FIPS is not (yet) compatible with versions above 14!** We will assume version `14.0.6` in the following.
3. Unpack the release archive and switch to the source directory (note that this will also create a directory called `cmake`):
    ```bash
    tar xvf llvm-14.0.6.src.tar.xz
    cd llvm-14.0.6.src/
    ```
4. Create a `build` folder, switch into it and run CMake:
    ```bash
    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=true -DCMAKE_INSTALL_PREFIX=/your/prefix/for/llvm
    ```
    Notes:
    - You can choose any empty folder as the install prefix for LLVM. In the following we assume that your normal user has write access to this folder.
    - We recommend enabling assertions as this can help us with debugging FIPS.
    - **If CMake aborts with an error about a non-existent directory for benchmarks:** This appears to be a bug in the build config of LLVM. You can fix it manually by going into `build/CMakeCache.txt` and turning every option that is related to benchmarks to `OFF`.
5. Build LLVM by running Make:
    ```bash
    make -j4
    make install
    ```
    Note: Building LLVM can take a long time and consume a considerable amount of resources. If your computer can handle it, you can increase the number of concurrent build jobs by changing `-j4` to higher numbers. **Watch your RAM usage!**

Usage
=====

Once you have installed LLVM, using FIPS is very simple: Just create a new Rust project and add the crate `fips-md` to the list of dependencies in your `Cargo.toml`. FIPS is used as a library from within a normal Rust program. To make sure that FIPS finds your LLVM installation, you will have to set the following environment variable:
```bash
export LLVM_SYS_140_PREFIX=/your/prefix/for/llvm
```
If you are new to the Rust programming language, make sure to check out the excellent [Rust Book](https://doc.rust-lang.org/book/) for an introduction to the world of Rust.

Questions
=========

Here are answers to some questions you might have:

How does FIPS work?
-------------------
FIPS is built as like a compiler, i.e., it parses particle types, particle interactions and simulation schemes you define in a domain-specific language, performs static analysis (e.g., to determine where to place synchronization barriers) and then generates LLVM IR code that is fed into the JIT engine of LLVM. It also contains implementations of common acceleration structures that are necessary to achieve reasonable performance levels.

How stable is FIPS?
-------------------
As a software project, FIPS is still very young. While it technically allows you to run nearly any particle simulation you can imagine by building on top of Rust[^1], we still think there is much potential for improvement to the usability of FIPS. To not restrict the future development, we consider the API of FIPS (which includes the DSL) to be preliminary. Until version 1.0 of FIPS, backwards compatibility will be secondary to usability.

[^1]: FIPS allows you to modify the system state with Rust, so FIPS can by definition everything that Rust can.

Why "fearless"?
---------------
To exploit the full capabilities of modern hardware you usually need to try and parallelize computation-heavy applications (such as particle simulations). However, parallel programming is notoriously hard to get right and often introduces subtle bugs in the form of race conditions into your program. The situation is particularly bad when it comes to a special class of race conditions called *data races*[^2]. In (Safe) Rust, great care was taken to design the language in such a way that these errors cannot ever occur, thus allowing you to create parallel programs without fear of running into them. This "fearless concurrency" was a major inspiration for the creation of FIPS and as such we pay it homage in the name of the software.

[^2]: We use the same semantics as Rust here, i.e., a data race happens when two or more threads try to access the same memory location with one of the threads writing and without any synchronization in place. Check out the [corresponding section in the Rustonomicon](https://doc.rust-lang.org/nomicon/races.html) for more details. 

What about version 0.1 and 0.2?
-------------------------------
Designing programming languages is hard. To make matters worse, early mistakes tend to stick around for long times because of backwards compatibility (after all, breaking the code of other people in major ways is not exactly polite). We iterated through multiple designs before arriving at the first version we were satisfied with. To maintain consistency with our internal development environments, the version history of FIPS starts at version 0.3.