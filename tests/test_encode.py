import unittest

from compgnn.encode import parse_subtasks, prepare_records, resolve_column


class PreparationTests(unittest.TestCase):
    def test_filter_and_alignment(self):
        rows = [
            {"Task": "image", "original_task": None, "subtasks": None},
            {"Task": "QA", "original_task": "first", "subtasks": '["a", "b"]'},
            {"Task": "QA", "original_task": "second", "subtasks": []},
            {"Task": "QA", "original_task": "third", "subtasks": "['c']"},
        ]
        records, originals, subtasks, offsets = prepare_records(
            rows, "Task", "original_task", "subtasks")
        self.assertEqual([r["source_row"] for r in records], [1, 2, 3])
        self.assertEqual(originals, ["first", "second", "third"])
        self.assertEqual(subtasks, ["a", "b", "c"])
        self.assertEqual(offsets, [0, 2, 2, 3])
        self.assertEqual(len(prepare_records(rows, "Task", "original_task", "subtasks", 1)[0]), 1)

    def test_reject_bad_subtasks(self):
        for value in (None, "ordinary string", '["ok", 1]', '[""]', '{"a": "b"}'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_subtasks(value)

    def test_ambiguous_schema_requires_configuration(self):
        with self.assertRaises(ValueError):
            resolve_column(["a", "b"], None, ["a", "b"], "original")
        self.assertEqual(resolve_column(["a", "b"], "b", [], "original"), "b")

    def test_current_dataset_schema(self):
        columns = ["id", "original_task", "decomposition", "harm_index", "label", "task_type"]
        self.assertEqual(resolve_column(columns, None, ["Task", "task_type", "task"], "task"), "task_type")
        self.assertEqual(resolve_column(columns, None, ["decomposed_subtasks", "subtasks", "decomposition"], "subtasks"), "decomposition")


if __name__ == "__main__":
    unittest.main()
