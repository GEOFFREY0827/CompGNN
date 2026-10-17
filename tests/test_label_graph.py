import unittest

import numpy as np

from compgnn.experiments.archive.label_graph import infer_relation_edges


class LabelGraphTests(unittest.TestCase):
    def test_support_edges_preserve_order(self):
        edges = infer_relation_edges(np.eye(3), combine_threshold=0.99)
        support = [(edge["source"], edge["target"]) for edge in edges if edge["relation"] == "support"]
        self.assertEqual(support, [(0, 1), (1, 2)])

    def test_similar_nodes_get_combine_label(self):
        embeddings = np.asarray([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
        edges = infer_relation_edges(embeddings, combine_threshold=0.9)
        combine = [edge for edge in edges if edge["relation"] == "combine"]
        self.assertEqual((combine[0]["source"], combine[0]["target"]), (0, 1))
        self.assertEqual(combine[0]["source_rule"], "embedding_cosine_threshold")

    def test_empty_example_is_rejected(self):
        with self.assertRaises(ValueError):
            infer_relation_edges(np.empty((0, 4)))


if __name__ == "__main__":
    unittest.main()
