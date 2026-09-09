- Prompt Injection: System override instructions, fake autonomous refund commands.
- Slang & Typo Invariance: Phonetic spacing, heavy slang/abbreviations, all-caps yelling.
- Sarcasm & Rhetoric: Sarcastic praise over severe delays, ironic Prime benefits.
- Negative Boundaries: Negated order cancellation, negated return intent.

---

## 7. How to Reproduce All Results via Single Universal Entrypoint

The entire system is orchestrated via [`main.py`](main.py):

```bash
# 1. Start the Live Interactive Web Dashboard & REST Microservice (Default)
python main.py

# 2. Run Complete Headline Reproduction Benchmark (200 Golden Cases)
python main.py benchmark

# 3. Run Behavioral & Adversarial Stress Suite (12 Stress Checks)
python main.py stress

# 4. Run Complete 27-Test Unit & Integration Suite
python main.py test

# 5. Launch Terminal Interactive Shell
python main.py demo

# 6. Run Complete End-to-End Verification and Start Server
python main.py all
```
