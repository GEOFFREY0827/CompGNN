import unittest

import torch

from compgnn.experiments.latest.relation_retrieval import (
    Edge,
    RelationConditionedRetriever,
    extract_k_hop_subgraph,
)


class RelationRetrievalTests(unittest.TestCase):
    def test_relation_conditioned_forward_keeps_attention_and_raw_path(self):
        torch.manual_seed(4)
        model = RelationConditionedRetriever(6, ["combine", "support"], hidden_dim=8)
        query = torch.randn(6, requires_grad=True)
        nodes = torch.randn(5, 6)
        result = model(query, "combine", nodes, top_k=2)
        self.assertEqual(tuple(result.transformed_query.shape), (6,))
        self.assertEqual(tuple(result.attention.shape), (5,))
        self.assertEqual(tuple(result.selected.shape), (2,))
        self.assertAlmostEqual(float(result.attention.sum().detach()), 1.0, places=5)
        (result.attention @ result.scores).backward()
        self.assertIsNotNone(model.query_transform[0].weight.grad)

    def test_sequence_excludes_previous_nodes(self):
        torch.manual_seed(5)
        model = RelationConditionedRetriever(4, ["combine", "support"])
        steps = model.retrieve_sequence(torch.randn(4), ["combine", "support"], torch.randn(4, 4))
        self.assertEqual(len(steps), 2)
        self.assertTrue(set(steps[0].selected.tolist()).isdisjoint(steps[1].selected.tolist()))

    def test_relation_labelled_three_hop_extraction(self):
        edges = [
            Edge(0, 1, "combine"),
            Edge(1, 2, "support"),
            Edge(2, 3, "combine"),
            Edge(3, 4, "support"),
        ]
        nodes, selected = extract_k_hop_subgraph([0], edges, max_hops=3)
        self.assertEqual(nodes, [0, 1, 2, 3])
        self.assertEqual([edge.relation for edge in selected], ["combine", "support", "combine"])

    def test_relation_specific_hop_cost(self):
        edges = [Edge(0, 1, "combine"), Edge(1, 2, "support")]
        nodes, _ = extract_k_hop_subgraph([0], edges, max_hops=2, relation_hops={"combine": 1, "support": 2})
        self.assertEqual(nodes, [0, 1])

if __name__ == "__main__":
    unittest.main()
