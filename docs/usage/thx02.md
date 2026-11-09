# THX-02 decisions

THX-02 uses a joint decision pipeline rather than the text generation API.
The implementation and current verification limits are recorded in
[the validation matrix](../../tools/oracle/thx02/validation.md).

Use the existing executable with a converted model and a JSON request:

```sh
cargo run --locked --profile release-fast --bin rust-model-inference -- \
  --model /Users/gouzi/Documents/git/rust-model-inference/models/THX-02/THX-02-F32.gguf \
  --decision-request request.json --compute cpu --threads 4
```

`--decision-request -` reads the request from stdin. The result is one JSON
object on stdout. `--decision-thinker thinker.gguf` supplies an optional merged
thinker checkpoint; the request controls thinking. `--max-context` caps the
encoded request. Generation, media and other task flags conflict with this
mode; put native task inputs in the request instead.

```json
{
  "state": {"message": "The package has arrived."},
  "questions": {
    "intent": {
      "type": "choice",
      "instructions": "Choose the best intent.",
      "criteria": {"delivery": "Delivery update", "billing": "Billing question"}
    },
    "delivered": {
      "type": "noul",
      "instructions": "Has delivery completed?",
      "criteria": {"false": "Delivery is incomplete", "true": "Delivery has completed"}
    },
    "urgency": {
      "type": "score",
      "instructions": "Rate urgency.",
      "criteria": ["low", "medium", "high"]
    }
  },
  "images": [],
  "think": false,
  "tta": 1,
  "return_thoughts": false
}
```

`state` and instructions may contain JSON. Image strings may be local paths
or data URLs in CLI requests. The CLI validates image bytes and converts local
files to data URLs before constructing the model; CLI image URLs never trigger
network access. Thinking accepts `false`, `true` or `"auto"`;
TTA accepts 1–8. Choice probabilities use the supplied option names. Score
probabilities use zero-based level indices, with a legend and expected index.
Noul returns the probability of the true criterion.

Start the existing server with the same model:

```sh
cargo run --locked --profile release-fast --bin rust-model-server -- \
  --model /Users/gouzi/Documents/git/rust-model-inference/models/THX-02/THX-02-F32.gguf \
  --host 127.0.0.1 --port 8080 --threads 4

curl http://127.0.0.1:8080/v1/decide \
  -H 'Content-Type: application/json' --data-binary @request.json
```

The server selects the decision backend only for `general.architecture=thx02`.
It exposes `POST /v1/decide` and `POST /v1/decide/batch`; chat and completion
routes are absent. Batch requests contain `{"requests":[request1,request2]}`
and return `{"model":"THX-02","results":[result1,result2]}`. Every batch
item is validated before any model execution. The endpoints do not stream.
Both CLI and HTTP reject `stream` values other than `false`; omit the field
for ordinary requests.
Malformed requests and images return 400, concurrent model requests return
429, and model execution failures return 500.

HTTP image strings must be data URLs. Remote HTTP/HTTPS URLs are also
available with `--allow-remote-images`, under the shared download size,
timeout, redirect and public-address checks. HTTP requests cannot open local
files, including `file://` URLs.

The existing JEV input forms are also available:

```sh
cargo run --locked --profile release-fast --bin rust-model-inference -- \
  --model /Users/gouzi/Documents/git/rust-model-inference/models/THX-02/THX-02-F32.gguf \
  --jev --jev-context 'The package arrived.' \
  --jev-question 'Choose the topic' --jev-option delivery --jev-option billing \
  --jev-output json
```

Single, binary, numeric score, paired `--jev-multi`, and `--jev-block`
questions map to the same native decision engine. `--image path.png` supplies
an image without a separate `--mmproj`. Numeric JEV candidates retain their
original values when computing the expected score. `--jev-output json`
prints one result per line; text output prints the same results with
indentation. The server exposes the corresponding `/v1/jev/score`,
`/v1/jev/grouped`, `/v1/jev/image` and `/v1/jev/image_grouped` shapes. Image
routes use `image_url`; grouped routes use `mode: "multi_select"` or
`"block_choice"`. Model-calibrated probabilities are used directly; generation
temperature is not an input to these routes. JEV always uses `think:false`,
even when a thinker checkpoint is supplied.
