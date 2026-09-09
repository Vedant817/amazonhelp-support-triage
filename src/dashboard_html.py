"""
Modern Interactive Dashboard UI for @AmazonHelp Support AI Agent.
Served directly via FastAPI at / and /dashboard.
"""

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>@AmazonHelp AI Agent - Live Triage & Inspection Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #0f172a; color: #f8fafc; }
        .mono { font-family: 'JetBrains Mono', monospace; }
        .glow { box-shadow: 0 0 25px -5px rgba(56, 189, 248, 0.15); }
    </style>
</head>
<body class="min-h-screen p-4 md:p-8">
    <div class="max-w-7xl mx-auto space-y-6">
        <!-- Header -->
        <header class="flex flex-col md:flex-row justify-between items-start md:items-center bg-slate-900 border border-slate-800 rounded-2xl p-6 glow">
            <div>
                <div class="flex items-center gap-3">
                    <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-amber-500 to-orange-500 flex items-center justify-center font-bold text-slate-950 text-xl">a</div>
                    <div>
                        <h1 class="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
                            @AmazonHelp AI Support Agent
                            <span class="text-xs bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 px-2.5 py-0.5 rounded-full font-medium">Production Live</span>
                        </h1>
                        <p class="text-sm text-slate-400">Calibrated Multi-Task Triage • Split Conformal Guarantees • Hybrid RRF • Pre-Dispatch Guardrails</p>
                    </div>
                </div>
            </div>
            <div class="mt-4 md:mt-0 flex gap-3 text-xs mono">
                <div class="bg-slate-800 border border-slate-700 px-3 py-1.5 rounded-lg text-slate-300">
                    <span class="text-slate-500">Conformal Target:</span> 95.0%
                </div>
                <div class="bg-slate-800 border border-slate-700 px-3 py-1.5 rounded-lg text-slate-300">
                    <span class="text-slate-500">SLA Latency:</span> &lt;100ms
                </div>
                <a href="/docs" target="_blank" class="bg-indigo-600 hover:bg-indigo-500 text-white px-3 py-1.5 rounded-lg font-medium transition flex items-center gap-1">
                    API Docs &rarr;
                </a>
            </div>
        </header>

        <!-- Main Workspace -->
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <!-- Left Panel: Input & Quick Presets -->
            <div class="lg:col-span-5 space-y-4">
                <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
                    <h2 class="text-base font-semibold text-slate-200 flex items-center gap-2">
                        <span>💬</span> Incoming Customer Tweet
                    </h2>
                    
                    <textarea id="tweetInput" rows="4" class="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500 transition font-sans" placeholder="Type or paste a customer tweet here..."></textarea>
                    
                    <div class="flex justify-between items-center text-xs text-slate-400">
                        <span id="charCount">0 / 280 characters</span>
                        <button id="submitBtn" onclick="runTriage()" class="bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold px-5 py-2 rounded-xl transition flex items-center gap-2">
                            <span>Execute Triage</span>
                            <span id="spinner" class="hidden animate-spin">⏳</span>
                        </button>
                    </div>

                    <!-- Preset Scenarios -->
                    <div class="pt-4 border-t border-slate-800 space-y-2">
                        <div class="text-xs font-medium text-slate-400">Quick Adversarial & Benchmark Presets:</div>
                        <div class="flex flex-wrap gap-2 text-xs">
                            <button onclick="setPreset(1)" class="bg-slate-800 hover:bg-slate-700 text-slate-300 px-2.5 py-1.5 rounded-lg border border-slate-700/50">Porch Piracy (Theft)</button>
                            <button onclick="setPreset(2)" class="bg-slate-800 hover:bg-slate-700 text-slate-300 px-2.5 py-1.5 rounded-lg border border-slate-700/50">Public PII Leak</button>
                            <button onclick="setPreset(3)" class="bg-slate-800 hover:bg-slate-700 text-slate-300 px-2.5 py-1.5 rounded-lg border border-slate-700/50">Sarcastic Praise Damage</button>
                            <button onclick="setPreset(4)" class="bg-slate-800 hover:bg-slate-700 text-slate-300 px-2.5 py-1.5 rounded-lg border border-slate-700/50">Negated Return Intent</button>
                            <button onclick="setPreset(5)" class="bg-slate-800 hover:bg-slate-700 text-slate-300 px-2.5 py-1.5 rounded-lg border border-slate-700/50">3P Seller Dispute</button>
                            <button onclick="setPreset(6)" class="bg-slate-800 hover:bg-slate-700 text-slate-300 px-2.5 py-1.5 rounded-lg border border-slate-700/50">Routine Transit Track</button>
                        </div>
                    </div>
                </div>

                <!-- Pipeline Telemetry -->
                <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-3 text-xs">
                    <h3 class="font-semibold text-slate-300 flex items-center justify-between">
                        <span>⚡ System Telemetry</span>
                        <span id="latencyBadge" class="mono text-cyan-400 bg-cyan-950/40 border border-cyan-800/40 px-2 py-0.5 rounded">Ready</span>
                    </h3>
                    <div class="space-y-1.5 mono text-slate-400">
                        <div class="flex justify-between"><span>Inference Engine:</span><span class="text-slate-200">Logit Bayesian N-Gram TF-IDF</span></div>
                        <div class="flex justify-between"><span>Retriever:</span><span class="text-slate-200">Okapi BM25 + Sublinear RRF (k=20)</span></div>
                        <div class="flex justify-between"><span>Coverage Form:</span><span class="text-slate-200">Finite-Sample Split Conformal</span></div>
                        <div class="flex justify-between"><span>Pre-Dispatch Guard:</span><span class="text-slate-200">17-Digit Redact + URL Whitelist</span></div>
                    </div>
                </div>
            </div>

            <!-- Right Panel: Full Triage Analysis -->
            <div class="lg:col-span-7 space-y-4">
                <!-- Triage Decision Card -->
                <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
                    <div class="flex justify-between items-start">
                        <div>
                            <span class="text-xs font-semibold text-slate-400 tracking-wider uppercase">Triage Routing Decision</span>
                            <div class="flex items-center gap-3 mt-1">
                                <span id="decisionBadge" class="text-lg font-bold px-3 py-1 rounded-xl bg-slate-800 text-slate-400">Awaiting Input</span>
                                <span id="intentBadge" class="text-xs font-semibold px-2.5 py-1 rounded-lg bg-slate-800 text-slate-300 mono">-</span>
                            </div>
                        </div>
                        <div class="text-right">
                            <span class="text-xs text-slate-400">Stated Causal Reason</span>
                            <div id="statedReason" class="text-sm font-semibold text-amber-400 mono mt-0.5">-</div>
                        </div>
                    </div>

                    <!-- Metrics Grid -->
                    <div class="grid grid-cols-3 gap-3 pt-3 border-t border-slate-800 text-xs">
                        <div class="bg-slate-950/60 p-3 rounded-xl border border-slate-800/60">
                            <span class="text-slate-500">Calibrated Confidence</span>
                            <div id="confidenceVal" class="text-base font-bold text-slate-200 mono mt-1">-</div>
                        </div>
                        <div class="bg-slate-950/60 p-3 rounded-xl border border-slate-800/60">
                            <span class="text-slate-500">Conformal Set Size</span>
                            <div id="conformalVal" class="text-base font-bold text-slate-200 mono mt-1">-</div>
                        </div>
                        <div class="bg-slate-950/60 p-3 rounded-xl border border-slate-800/60">
                            <span class="text-slate-500">OOD Energy Score</span>
                            <div id="oodVal" class="text-base font-bold text-slate-200 mono mt-1">-</div>
                        </div>
                    </div>

                    <!-- Dispatched Reply Box -->
                    <div class="space-y-2 pt-2">
                        <div class="flex justify-between items-center text-xs">
                            <span class="font-semibold text-slate-300">🛡️ Dispatched Public Tweet Reply</span>
                            <span id="replyLength" class="mono text-slate-400">0 / 280 chars</span>
                        </div>
                        <div id="replyBox" class="bg-slate-950 border border-slate-800 rounded-xl p-4 text-sm text-slate-200 leading-relaxed min-h-[72px] font-sans flex items-center text-slate-500 italic">
                            Run triage to generate a grounded, policy-compliant reply...
                        </div>
                    </div>

                    <!-- Triggers & Details Accordion -->
                    <div class="space-y-2 pt-2 border-t border-slate-800 text-xs">
                        <div class="text-slate-400 font-medium">Policy Verifications & Triggers:</div>
                        <div id="triggersList" class="flex flex-wrap gap-1.5 text-slate-400">
                            <span class="bg-slate-800/50 px-2 py-1 rounded text-slate-500">None</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const presets = {
            1: "@AmazonHelp my package was marked delivered but a porch pirate stole it from my front door!",
            2: "Help! Someone hacked my prime account, my email is victim@example.com and order is 112-8291029-3829102!",
            3: "Oh wonderful! Amazon delivery driver threw my glass lamp over a 6ft fence! Outstanding service!",
            4: "I don't want to return this, I love the shoes, but you sent size 10 instead of size 8.",
            5: "The third party marketplace seller refuses to reply or send my order after 4 days!",
            6: "@AmazonHelp where is my package? The tracking says out for delivery today."
        };

        function setPreset(idx) {
            document.getElementById('tweetInput').value = presets[idx];
            updateCharCount();
            runTriage();
        }

        function updateCharCount() {
            const txt = document.getElementById('tweetInput').value;
            document.getElementById('charCount').innerText = `${txt.length} / 280 characters`;
        }

        document.getElementById('tweetInput').addEventListener('input', updateCharCount);

        async function runTriage() {
            const text = document.getElementById('tweetInput').value.trim();
            if (!text) return;

            document.getElementById('spinner').classList.remove('hidden');
            document.getElementById('submitBtn').disabled = true;

            try {
                const res = await fetch('/api/v1/triage', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text })
                });

                if (!res.ok) throw new Error('API request failed');
                const data = await res.json();

                // 1. Update Decision Badge
                const decBadge = document.getElementById('decisionBadge');
                if (data.escalation.should_escalate) {
                    decBadge.className = "text-lg font-bold px-3 py-1 rounded-xl bg-amber-500/20 text-amber-400 border border-amber-500/30";
                    decBadge.innerText = "🚨 ESCALATE TO HUMAN";
                } else {
                    decBadge.className = "text-lg font-bold px-3 py-1 rounded-xl bg-emerald-500/20 text-emerald-400 border border-emerald-500/30";
                    decBadge.innerText = "🤖 AUTO-HANDLED";
                }

                // 2. Intent Badge
                document.getElementById('intentBadge').innerText = data.intent;

                // 3. Stated Reason
                document.getElementById('statedReason').innerText = data.escalation.stated_reason;

                // 4. Metrics
                document.getElementById('confidenceVal').innerText = (data.confidence * 100).toFixed(1) + "%";
                document.getElementById('conformalVal').innerText = `${data.conformal_prediction.set_size} (${data.conformal_prediction.prediction_set.join(', ')})`;
                document.getElementById('oodVal').innerText = `${data.ood_detection.energy_score} (${data.ood_detection.status})`;

                // 5. Reply Box
                const replyBox = document.getElementById('replyBox');
                replyBox.className = "bg-slate-950 border border-slate-800 rounded-xl p-4 text-sm text-slate-100 leading-relaxed min-h-[72px] font-sans";
                replyBox.innerText = data.draft_reply;
                document.getElementById('replyLength').innerText = `${data.character_count} / 280 chars`;

                // 6. Triggers
                const trigContainer = document.getElementById('triggersList');
                trigContainer.innerHTML = '';
                (data.escalation.triggers || []).forEach(t => {
                    const span = document.createElement('span');
                    span.className = "bg-slate-800 px-2 py-0.5 rounded text-amber-300 mono text-xs";
                    span.innerText = t;
                    trigContainer.appendChild(span);
                });

                // 7. Latency
                document.getElementById('latencyBadge').innerText = `${data.latency_ms} ms`;

            } catch (err) {
                alert('Error running triage: ' + err.message);
            } finally {
                document.getElementById('spinner').classList.add('hidden');
                document.getElementById('submitBtn').disabled = false;
            }
        }
    </script>
</body>
</html>
"""
