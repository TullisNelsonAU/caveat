# probdisasm
This is still a major work in progress.

### Background
probdisasm is a disassembler based on two concepts: superset disassembly and probabilistic analysis. Superset disassembly attempts to decode an instruction for each byte or offset in the binary or executable section. Probabilistic analysis then refines the results by assigning a probability to each decoded instruction based on "hints" such as control flow patterns and data flow edges. The goal for this repository is to implement the approach specified in the paper, "Probabilistic disassembly" by Miller et al and to further extend it to support new static and machine learning based hints.


#### Why not the original artifact
We found that the implementation did not fully explore or implement the algorithm from the paper. We have tried integrating ML and other probalistic methods in the past, but with the lack of activity in the BAP ecosphere and the non probalisitc based engine in the BAP plugin we felt that we and the community could benefit for a more accessible version of the idea for future research.

### Installation
#### Rust
```bash
cargo add probdisasm
```
#### Python
```bash
uv add probdisasm 
```
### Usage


### Credit

This work is based heavily on the following paper.
```
@inproceedings{10.1109/ICSE.2019.00121,
author = {Miller, Kenneth and Kwon, Yonghwi and Sun, Yi and Zhang, Zhuo and Zhang, Xiangyu and Lin, Zhiqiang},
title = {Probabilistic disassembly},
year = {2019},
publisher = {IEEE Press},
url = {https://doi.org/10.1109/ICSE.2019.00121},
doi = {10.1109/ICSE.2019.00121},
booktitle = {Proceedings of the 41st International Conference on Software Engineering},
location = {Montreal, Quebec, Canada},
series = {ICSE '19}
}
```
