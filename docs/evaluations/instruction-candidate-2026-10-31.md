# Model evaluation review

Accuracy: **57/170 (33.5%)**.
This measures the recorded tasks; it does not establish production readiness.

Checkpoint SHA-256: `50664e4e548b9f3d1b4e80be008ad36488aa8c5f136663dd31f1e3171198ac70`
Suite SHA-256: `b2c1c3737b03d6d2a2711e7fc36044575e1e35b930d26588da76b79208e70df0`

## Results by task type

| Type | Correct / tasks | Accuracy | Wilson 95% interval |
|---|---:|---:|---:|
| code | 0/18 | 0.0% | 0.0%–17.6% |
| generation | 2/16 | 12.5% | 3.5%–36.0% |
| multiple\_choice | 55/136 | 40.4% | 32.6%–48.8% |

MC random-guess reference: 25.0%; compare with MC-only accuracy above.

Intervals describe uncertainty on these small task samples, not performance on all real-world work.

## Categories

| Category | Correct / tasks | Accuracy |
|---|---:|---:|
| ai\_security | 11/22 | 50.0% |
| cloud | 10/20 | 50.0% |
| code | 0/18 | 0.0% |
| generation | 2/16 | 12.5% |
| reasoning | 6/14 | 42.9% |
| secure\_development | 2/12 | 16.7% |
| security | 19/45 | 42.2% |
| software | 7/23 | 30.4% |

## Regression comparison

FAIL — regression or incompatible/incomplete evidence.

| Check | Outcome | Note |
|---|---|---|
| suite\_sha256 | PASS | evaluation provenance must match |
| heldout\_sha256 | PASS | evaluation provenance must match |
| scoring\_version | PASS | evaluation provenance must match |
| code\_evaluator | PASS | evaluation provenance must match |
| complete\_suite | PASS | full suite required; smoke runs cannot pass |
| perplexity | SKIP | not enforced |
| bits\_per\_byte | PASS |  |
| top1\_acc | SKIP | not enforced |
| top5\_acc | SKIP | not enforced |
| overall\_acc | PASS |  |
| category:ai\_security | PASS | category regression floor |
| category:cloud | PASS | category regression floor |
| category:code | PASS | category regression floor |
| category:generation | PASS | category regression floor |
| category:reasoning | FAIL | category regression floor |
| category:secure\_development | PASS | category regression floor |
| category:security | PASS | category regression floor |
| category:software | PASS | category regression floor |
| code\_execution | PASS | all code tasks require an available independent evaluator |

## Failure triage

| Task | Category | Type | Evidence |
|---|---|---|---|
| sec-sqli | security | multiple\_choice | chose 3, answer 0 |
| sec-xss | security | multiple\_choice | chose 3, answer 1 |
| sec-pwstore | security | multiple\_choice | chose 0, answer 3 |
| sec-leak | security | multiple\_choice | chose 2, answer 1 |
| sec-leastpriv | security | multiple\_choice | chose 3, answer 1 |
| sec-csrf | security | multiple\_choice | chose 2, answer 0 |
| sec-saltwhy | security | multiple\_choice | chose 2, answer 1 |
| sec-defexploit | security | multiple\_choice | chose 0, answer 1 |
| sec-mfa | security | multiple\_choice | chose 1, answer 0 |
| sec-logging | security | multiple\_choice | chose 0, answer 1 |
| sec-idor | security | multiple\_choice | chose 2, answer 0 |
| sec-zerotrust | security | multiple\_choice | chose 0, answer 1 |
| sec-zta2 | security | multiple\_choice | chose 3, answer 0 |
| sec-supplychain | security | multiple\_choice | chose 3, answer 0 |
| sec-vpn | security | multiple\_choice | chose 2, answer 0 |
| sec-pentest | security | multiple\_choice | chose 2, answer 0 |
| sec2-tls | security | multiple\_choice | chose 3, answer 1 |
| sec2-sqli | security | multiple\_choice | chose 0, answer 1 |
| sec2-xss | security | multiple\_choice | chose 1, answer 0 |
| sec2-csrf | security | multiple\_choice | chose 3, answer 0 |
| sec2-zerotrust | security | multiple\_choice | chose 2, answer 1 |
| sec2-segmentation | security | multiple\_choice | chose 2, answer 1 |
| sec2-patch | security | multiple\_choice | chose 3, answer 0 |
| sec2-ir | security | multiple\_choice | chose 3, answer 1 |
| sec2-secretsmgmt | security | multiple\_choice | chose 0, answer 1 |
| sec2-defense | security | multiple\_choice | chose 3, answer 1 |
| sw-bigo | software | multiple\_choice | chose 1, answer 0 |
| sw-idempotent | software | multiple\_choice | chose 3, answer 0 |
| sw-get | software | multiple\_choice | chose 1, answer 0 |
| sw-dry | software | multiple\_choice | chose 1, answer 0 |
| sw-git | software | multiple\_choice | chose 1, answer 0 |
| sw-stack | software | multiple\_choice | chose 2, answer 0 |
| sw-index | software | multiple\_choice | chose 2, answer 0 |
| sw-hashmap | software | multiple\_choice | chose 3, answer 0 |
| sw-recursion | software | multiple\_choice | chose 1, answer 0 |
| sw-null | software | multiple\_choice | chose 3, answer 0 |
| sw2-vcs | software | multiple\_choice | chose 3, answer 1 |
| sw2-pure | software | multiple\_choice | chose 0, answer 1 |
| sw2-bigo | software | multiple\_choice | chose 0, answer 1 |
| sw2-pin | software | multiple\_choice | chose 2, answer 0 |
| sw2-immutable | software | multiple\_choice | chose 0, answer 1 |
| sw2-test | software | multiple\_choice | chose 3, answer 1 |
| cld-container | cloud | multiple\_choice | chose 2, answer 0 |
| cld-k8s | cloud | multiple\_choice | chose 2, answer 0 |
| cld-secrets | cloud | multiple\_choice | chose 1, answer 0 |
| cld-leastpriv | cloud | multiple\_choice | chose 1, answer 0 |
| cld-misconfig | cloud | multiple\_choice | chose 2, answer 0 |
| cld-dast | cloud | multiple\_choice | chose 1, answer 0 |
| cld-obs | cloud | multiple\_choice | chose 3, answer 0 |
| cloud2-atrest | cloud | multiple\_choice | chose 0, answer 1 |
| cloud2-sg | cloud | multiple\_choice | chose 3, answer 1 |
| cloud2-private | cloud | multiple\_choice | chose 3, answer 1 |
| ai-injection | ai\_security | multiple\_choice | chose 1, answer 0 |
| ai-untrusted | ai\_security | multiple\_choice | chose 1, answer 0 |
| ai-humanloop | ai\_security | multiple\_choice | chose 1, answer 0 |
| aisec2-direct | ai\_security | multiple\_choice | chose 1, answer 0 |
| aisec2-untrusted | ai\_security | multiple\_choice | chose 2, answer 1 |
| aisec2-tenant | ai\_security | multiple\_choice | chose 0, answer 1 |
| aisec2-rag | ai\_security | multiple\_choice | chose 0, answer 1 |
| aisec2-secrets | ai\_security | multiple\_choice | chose 0, answer 1 |
| aisec2-output | ai\_security | multiple\_choice | chose 0, answer 1 |
| aisec2-sandbox | ai\_security | multiple\_choice | chose 2, answer 1 |
| aisec2-promptleak | ai\_security | multiple\_choice | chose 3, answer 0 |
| rsn-highrisk | reasoning | multiple\_choice | chose 2, answer 0 |
| rsn-citation | reasoning | multiple\_choice | chose 3, answer 0 |
| rsn-injection | reasoning | multiple\_choice | chose 3, answer 0 |
| rsn-regress | reasoning | multiple\_choice | chose 2, answer 0 |
| rsn-authz-text | reasoning | multiple\_choice | chose 2, answer 0 |
| rz2-counting | reasoning | multiple\_choice | chose 2, answer 1 |
| rz2-necessary | reasoning | multiple\_choice | chose 0, answer 1 |
| rz2-setdiff | reasoning | multiple\_choice | chose 3, answer 1 |
| gen-private | generation | generation | 'certificate issued by the CA.\\n\\n   A' |
| gen-https | generation | generation | 'version 1.0 (but the\\n   version ' |
| gen-encrypt | generation | generation | 'a data that is\\n      transmitted in' |
| gen-firewall | generation | generation | 'working blocking or working' |
| gen-password | generation | generation | 'cookie with a cookie with ' |
| gen-digit | generation | generation | 'of which is the ' |
| gen-sql | generation | generation | 'a "core" core codes' |
| gen-token | generation | generation | 'a JWS Set\\n      \[RFC7519' |
| gen-tls13 | generation | generation | '1.0 \[RFC4346\]' |
| gen-cve | generation | generation | 'or a publicly release of' |
| gen2-tls | generation | generation | 'in a\\n      manner or in a' |
| gen2-encrypt | generation | generation | 'in the block cipher text.\\n\\n   ' |
| gen2-firewall | generation | generation | 'wireless LAN or IPsec' |
| gen2-auth | generation | generation | 'in a\\n      certification path. (See:' |
| code-add | code | code | worker exited unsuccessfully |
| code-max | code | code | worker exited unsuccessfully |
| code-rev | code | code | worker exited unsuccessfully |
| code-even | code | code | worker exited unsuccessfully |
| code-count | code | code | worker exited unsuccessfully |
| code-fib | code | code | worker exited unsuccessfully |
| code-palin | code | code | worker exited unsuccessfully |
| code-sum | code | code | worker exited unsuccessfully |
| code2-even | code | code | worker exited unsuccessfully |
| code2-reverse | code | code | worker exited unsuccessfully |
| code2-vowels | code | code | worker exited unsuccessfully |
| code2-factorial | code | code | worker exited unsuccessfully |
| code2-palindrome | code | code | worker exited unsuccessfully |
| code2-gcd | code | code | worker exited unsuccessfully |
| code2-sumlist | code | code | worker exited unsuccessfully |
| code2-c2f | code | code | worker exited unsuccessfully |
| code2-unique | code | code | worker exited unsuccessfully |
| code2-fizz | code | code | worker exited unsuccessfully |
| secure-dev-sqli | secure\_development | multiple\_choice | chose 3, answer 0 |
| secure-dev-shell | secure\_development | multiple\_choice | chose 0, answer 1 |
| secure-dev-tenant | secure\_development | multiple\_choice | chose 0, answer 2 |
| secure-dev-injection | secure\_development | multiple\_choice | chose 0, answer 3 |
| secure-dev-tls | secure\_development | multiple\_choice | chose 2, answer 0 |
| secure-dev-redirect | secure\_development | multiple\_choice | chose 0, answer 1 |
| secure-dev-revocation | secure\_development | multiple\_choice | chose 3, answer 2 |
| secure-dev-path | secure\_development | multiple\_choice | chose 3, answer 0 |
| secure-dev-artifact | secure\_development | multiple\_choice | chose 1, answer 2 |
| secure-dev-overload | secure\_development | multiple\_choice | chose 2, answer 3 |

Model weights were not changed by this review.
