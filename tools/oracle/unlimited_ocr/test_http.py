import json
import unittest

from check_http import check_chat_stream


def stream(*frames):
    return "\n\n".join("data: " + json.dumps(frame) for frame in frames) + "\n\ndata: [DONE]\n"


class HttpCheckerTests(unittest.TestCase):
    token = {"choices": [{"delta": {"content": "<PAGE>"}, "finish_reason": None}]}
    finished = {"choices": [{"delta": {}, "finish_reason": "length"}]}

    def test_complete_one_token_stream(self):
        check_chat_stream(stream(self.token, self.finished))

    def test_error_after_expected_output_is_a_failure(self):
        with self.assertRaises(AssertionError):
            check_chat_stream(stream(self.token, {"error": {"message": "worker failed"}}))

    def test_done_without_terminal_finish_is_a_failure(self):
        with self.assertRaises(AssertionError):
            check_chat_stream(stream(self.token))


if __name__ == "__main__":
    unittest.main()
