# UniG-Encoder
UniG-Encoder: A Universal Feature Encoder for Graph and Hypergraph Node Classification.

# Abstract
Despite the decent performance and fruitful applications of Graph Neural Networks (GNNs), Hypergraph Neural Networks (HGNNs), and their well-designed variants, on some commonly used benchmark graphs and hypergraphs, they are outperformed by even a simple Multi-Layer Perceptron. This observation motivates a reexamination of the design paradigm of the current GNNs and HGNNs and poses challenges of extracting graph features effectively. In this work, a universal feature encoder for both graph and hypergraph representation learning is designed, called UniG-Encoder. The architecture starts with a forward transformation of the topological relationships of connected nodes into edge or hyperedge features via a normalized projection matrix. The resulting edge/hyperedge features, together with the original node features, are fed into a neural network. The encoded node embeddings are then derived from the reversed transformation, described by the transpose of the projection matrix, of the network’s output, which can be further used for tasks such as node classification. The designed projection matrix, encoding the graph features, is intuitive and interpretable. The proposed architecture, in contrast to the traditional spectral-based and/or message passing approaches, simultaneously and comprehensively exploits the node features and graph/hypergraph topologies in an efficient and unified manner, covering both heterophilic and homophilic graphs. Furthermore, a variant version, UniG-Encoder II, is devised to leverage multi-hop node information. Extensive experiments are conducted and demonstrate the superior performance of the proposed framework on twelve representative hypergraph datasets and six real-world graph datasets, compared to the state-of-the-art methods.

![公式对比](formula.png)

# Citation
```
@article{zou2024unig,
  title={Unig-encoder: A universal feature encoder for graph and hypergraph node classification},
  author={Zou, Minhao and Gan, Zhongxue and Wang, Yutong and Zhang, Junheng and Sui, Dongyan and Guan, Chun and Leng, Siyang},
  journal={Pattern Recognition},
  volume={147},
  pages={110115},
  year={2024},
  publisher={Elsevier}
}
```
