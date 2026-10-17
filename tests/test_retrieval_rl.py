import unittest

import torch

from compgnn.experiments.latest.relation_retrieval import Edge, RelationConditionedRetriever
from compgnn.experiments.latest.retrieval_rl import RetrievalRLExample, run_rl_epoch, score_node_action


class RetrievalRLTests(unittest.TestCase):
    def test_reward_distinguishes_correct_and_irrelevant_nodes(self):
        edges = [Edge(0, 1, "combine")]
        correct = score_node_action(1, 1, 0, "combine", edges)
        wrong = score_node_action(3, 1, 0, "combine", edges)
        self.assertGreater(correct.total, wrong.total)
        self.assertEqual(correct.relation, 0.5)
        self.assertEqual(wrong.irrelevant, -1.0)

    def test_reinforce_episode_logs_reward_components(self):
        torch.manual_seed(9)
        model = RelationConditionedRetriever(4, ["combine", "support"])
        example = RetrievalRLExample(
            query=torch.randn(4),
            candidate_nodes=torch.randn(4, 4),
            relations=["combine", "support"],
            gold_nodes=[1, 2],
            edges=[Edge(1, 2, "support")],
            target_intent=torch.randn(4),
            target_group=[1, 2],
            defense_scorer=lambda selected: 1.0,
        )
        stats = run_rl_epoch(model, [example], torch.optim.Adam(model.parameters(), lr=1e-3))
        self.assertIn("relation_reward", stats)
        self.assertIn("irrelevant_penalty", stats)
        self.assertIn("intent_reward", stats)
        self.assertIn("group_reward", stats)
        self.assertIn("defense_reward", stats)


if __name__ == "__main__":
    unittest.main()
