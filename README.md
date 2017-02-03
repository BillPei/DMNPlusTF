# Dynamic Memory Network Plus in Tensorflow

This is my implementaion of the super cool paper by Caiming Xiong, Stephen Merity, Richard Socher of MetaMind:
[DMN for Visual and Textual Question Answering](https://arxiv.org/abs/1603.01417)

Many many thanks to [@therne](https://github.com/therne/dmn-tensorflow) and [@barronalex](https://github.com/barronalex/Dynamic-Memory-Networks-in-TensorFlow) for sharing their code - very useful reference.

Gratitude to Richard Socher for sharing knowledge openly via cs224d and Andrej Karpathy (@karpathy) via cs231n!

![DMN Plus](https://github.com/rsethur/DMNPlusTF/raw/master/images/dmn_plus.png)

##Benchmarks
Test Error on the Facebook's babi 10k data set for Question Answering:

|    | Task                                           | This implementation | Xiong et al DMN+ |
|----|------------------------------------------------|-------------------|------------------|
| 1  | Basic   factoid QA with single supporting fact | 0                 | 0                |
| 2  | Factoid   QA with two supporting facts         | 0                 | 0.3              |
| 3  | Factoid   QA with three supporting facts       | 8.04              | 1.1              |
| 4  | Two   argument relations: subject vs. object   | 0                 | 0                |
| 5  | Three   argument relations                     | 0.23              | 0.5              |
| 6  | Yes/No   questions                             | 0                 | 0                |
| 7  | Counting                                       | 4.36              | 2.4              |
| 8  | Lists/Sets                                     | 0                 | 0                |
| 9  | Simple   Negation                              | 0                 | 0                |
| 10 | Indefinite   Knowledge                         | 0                 | 0                |
| 11 | Basic   coreference                            | 0                 | 0                |
| 12 | Conjunction                                    | 0                 | 0                |
| 13 | Compound   coreference                         | 0                 | 0                |
| 14 | Time   manipulation                            | 0.23              | 0.2              |
| 15 | Basic   deduction                              | 0                 | 0                |
| 16 | Basic   induction                              | 42.75             | 45.3             |
| 17 | Positional   reasoning                         | 3.46              | 4.2              |
| 18 | Reasoning   about size                         | 0.67              | 2.1              |


