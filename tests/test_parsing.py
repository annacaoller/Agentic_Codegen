import unittest

from agentic_codegen.parsing import parse_action, ActionParseError


class TestParsing(unittest.TestCase):
    def test_parse_action_success(self):
        txt = """\
hello
```action
{"tool_name":"write_code","args":{"x":1}}
```
"""
        action = parse_action(txt)
        self.assertEqual(action.tool_name, "write_code")
        self.assertEqual(action.args["x"], 1)

    def test_parse_action_missing_block(self):
        with self.assertRaises(ActionParseError):
            parse_action("no action here")

    def test_parse_action_invalid_json(self):
        txt = "```action\n{not json}\n```"
        with self.assertRaises(ActionParseError):
            parse_action(txt)


if __name__ == "__main__":
    unittest.main()