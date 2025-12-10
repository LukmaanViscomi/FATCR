# ui_faq.py
import streamlit as st


def render_faq() -> None:
    st.header("FAQ – How FACTR scores claims")

    # 1. What is FACTR?
    with st.expander("1. What is FACTR and what problem does it solve?"):
        st.markdown(
            """
FACTR (**Fact-based AI Checker for Truth & Reliability**) is a prototype tool for
**analysing theological claims in Muslim–Christian debates**.

It:

- takes a debate (e.g. a YouTube video),
- extracts **individual claims** made by the speakers,
- looks up relevant **Islamic** and **Christian** sources in a curated knowledge base,
- uses AI (**Natural Language Inference**, NLI) to decide whether those sources
  **agree** or **disagree** with each claim, and
- shows you the result as coloured verdicts plus the **actual passages** it used.

FACTR is a **decision-support tool** for scholars, students and researchers. It is
not a replacement for qualified religious scholarship.
"""
        )

    # 2. What sources does it use?
    with st.expander("2. What sources does FACTR use?"):
        st.markdown(
            """
FACTR works over a **curated, read-only knowledge base**.

At this stage the KB includes:

- **Islamic sources**
  - English translations of the **Qur’an**
  - Selected **ḥadīth** collections (in English)
  - A small set of widely used **tafsīr** (Qur’anic exegesis)
  - Other orthodox commentary where available

- **Christian sources**
  - English translations of the **Bible**
  - Representative **commentaries** on key doctrinal passages

Each passage is stored with metadata such as:
`tradition` (Islam/Christian), `source`, `book`, `chapter`, `verse`, etc.

⚠️ **Important:**  
The verdicts reflect **this curated KB**, not “Islam” or “Christianity” in their
entirety. If a school, denomination or commentary is missing, that will be
visible in the evidence list.
"""
        )

    # 3. How to read verdicts (per tradition + combined)
    with st.expander("3. How do I read the coloured verdicts?"):
        st.markdown(
            """
For **each claim**, FACTR gives **one verdict per tradition** (Islam / Christianity)
and **one combined verdict**.

### Per-tradition verdicts (Islam or Christianity)

These are shown as coloured pills in the claims table:

- 🟢 **Agrees**  
  The main sources in this tradition **clearly back the claim**.  
  → internally: strong “support” signal.

- 🔴 **Disagrees**  
  The main sources **clearly teach the opposite** of the claim.  
  → internally: strong “contradiction” signal.

- 🟧 **Divided**  
  Within this tradition there is **strong evidence on both sides**.  
  → internally: strong support **and** strong contradiction from different passages.

- ⚪ **Insufficient**  
  The retrieved passages are **too weak, ambiguous or off-topic** to decide.  
  → internally: no strong support or contradiction above stricter thresholds.

These verdicts say **how the curated sources behave w.r.t. the exact wording of the claim**,
not whether the tradition is “right” or “wrong” in an absolute sense.

### Combined verdict (both traditions)

Under **“Both sources”**, FACTR shows one summary verdict:

- 🟢 **Agreement**  
  At least one tradition **Agrees**, and **neither** clearly **Disagrees**.

- 🟣 **Doubtful**  
  At least one tradition **Disagrees**, and **neither** clearly **Agrees**.

- 🔴 **Conflicted**  
  One tradition **Agrees** and the other **Disagrees**  
  → explicit inter-faith clash on the claim.

- ⚪ **Insufficient**  
  Both sides are too weak, ambiguous or off-topic to say either way.

The combined verdict is just a **summary**. The main value is:
- the **per-tradition** verdicts, and
- the **actual passages** shown when you expand the evidence.
"""
        )

    # 4. Lay explanation of how scoring works
    with st.expander("4. In simple terms, how does FACTR decide these verdicts?"):
        st.markdown(
            """
For each claim:

1. **Find candidate evidence**  
   FACTR searches the knowledge base for the most relevant passages within:
   - Islamic sources, and
   - Christian sources.

2. **Ask an AI model how each passage relates to the claim**  
   For each passage, an NLI model is asked:
   - _Does this text **support** the claim?_
   - _Does it **contradict** the claim?_
   - _Or is it basically **irrelevant/neutral**?_

3. **Summarise per tradition**  
   For each tradition separately, FACTR looks at all these judgements and decides:

   - If strong support dominates → 🟢 **Agrees**
   - If strong contradiction dominates → 🔴 **Disagrees**
   - If there is strong support **and** strong contradiction → 🟧 **Divided**
   - If signals are too weak / noisy → ⚪ **Insufficient**

4. **Summarise both traditions together**  
   Finally, FACTR combines the two per-tradition verdicts into one overall pill:
   **Agreement**, **Doubtful**, **Conflicted** or **Insufficient**.

The labels are deliberately phrased as “Agrees / Disagrees” because testers
(including practising Muslim debaters) found these terms more intuitive than
“supported / contradicted”.
"""
        )

    # 5. Simple example (lay)
    with st.expander("5. Verdict - Example (non-technical)"):
        st.markdown(
            """
**Claim (from a debate):**  
> *“The Qur’an teaches that Jesus was not crucified.”*

**Step 1 – Retrieval**

- In the **Islamic** KB, FACTR finds verses like **Qur’an 4:157–158** and
  tafsīr that explicitly discuss the crucifixion.
- In the **Christian** KB, FACTR finds Gospel crucifixion passages
  (e.g. Mark 15, John 19) and commentaries that affirm the crucifixion.

**Step 2 – Passage-level judgements**

For each passage, the NLI model decides whether it **supports** or
**contradicts** the claim:

- Several Islamic verses and tafsīr → “this **supports** the claim”  
- Several Christian passages → “this **contradicts** the claim”

**Step 3 – Per-tradition verdicts**

- For **Islam**, strong support dominates → 🟢 **Agrees**  
  (“Islamic sources agree the Qur’an denies the crucifixion.”)

- For **Christianity**, strong contradiction dominates → 🔴 **Disagrees**  
  (“Christian sources disagree and affirm the crucifixion.”)

**Step 4 – Combined verdict**

- One tradition **Agrees** and the other **Disagrees** → 🔴 **Conflicted**

So the UI might show:

- Islam: 🟢 Agrees  
- Christianity: 🔴 Disagrees  
- Both sources: 🔴 Conflicted

You can then expand the claim to see exactly which verses and commentaries were used.
"""
        )

    # 6. Technical details (NLI, thresholds, top-K)
    with st.expander("6. Technical details – NLI, thresholds and aggregation"):
        st.markdown(
            r"""
Under the hood, FACTR uses a standard **Natural Language Inference (NLI)** model.

For each claim \(c\) and retrieved passage \(p\), the NLI model outputs three
probabilities:

- \(p_E\): the passage **entails** (supports) the claim  
- \(p_C\): the passage **contradicts** the claim  
- \(p_N\): the passage is **neutral/irrelevant**

(These three values sum to 1.)

### 6.1 Top-K retrieval per tradition

For each tradition \(t \in \{\text{Islam}, \text{Christianity}\}\):

1. Encode the claim with an embedding model.
2. Retrieve the **top K** passages from the KB where `tradition == t`
   (K is usually 10–20).
3. For each passage \(p_i\), compute \((p_E^{(i)}, p_C^{(i)}, p_N^{(i)})\).

### 6.2 Aggregating signals

For each tradition \(t\), FACTR computes:

- \(\text{support\_max}_t = \max_i p_E^{(i)}\)  
- \(\text{contradict\_max}_t = \max_i p_C^{(i)}\)  
- \(\text{support\_ratio}_t = \frac{1}{K} \sum_i \mathbb{1}\{p_E^{(i)} \ge \tau_E\}\)  
- \(\text{contradict\_ratio}_t = \frac{1}{K} \sum_i \mathbb{1}\{p_C^{(i)} \ge \tau_C\}\)

where \(\tau_E\) and \(\tau_C\) are strict thresholds (e.g. ~0.65–0.70), and \(\mathbb{1}\) is an indicator.

### 6.3 Mapping to per-tradition verdicts

Using these values, FACTR assigns one internal label per tradition:

- **supports** (→ 🟢 Agrees) if:
  - \(\text{support\_max}_t \ge \tau_E\), and  
  - \(\text{support\_max}_t - \text{contradict\_max}_t \ge \delta\)

- **contradicts** (→ 🔴 Disagrees) if:
  - \(\text{contradict\_max}_t \ge \tau_C\), and  
  - \(\text{contradict\_max}_t - \text{support\_max}_t \ge \delta\)

- **mixed** (→ 🟧 Divided) if both support and contradiction are strong,
  e.g. \(\text{support\_ratio}_t\) and \(\text{contradict\_ratio}_t\) both above a set fraction.

- **insufficient** (→ ⚪ Insufficient) otherwise.

Here \(\delta\) is a safety margin to avoid “Agrees” or “Disagrees”
when the model is nearly 50–50.

The UI pills are just a **human-friendly mapping** of these internal labels.
"""
        )

    # 7. Full technical worked example
    with st.expander("7. Verdict - Full technical worked example"):
        st.markdown(
            r"""
Consider the claim:

> *“The Qur’an denies that Jesus died on the cross.”*

### 7.1 Islamic side

Top retrieved passages (simplified):

- \(p_1\): Qur’an 4:157–158  
- \(p_2\): Tafsīr on 4:157  
- \(p_3\): Another verse mentioning Jesus

NLI outputs (illustrative):

- \(p_1\): \(p_E = 0.88\), \(p_C = 0.03\), \(p_N = 0.09\) → supports  
- \(p_2\): \(p_E = 0.81\), \(p_C = 0.05\), \(p_N = 0.14\) → supports  
- \(p_3\): \(p_E = 0.22\), \(p_C = 0.10\), \(p_N = 0.68\) → neutral

with thresholds \(\tau_E = 0.70\), \(\tau_C = 0.70\), margin \(\delta = 0.10\).

Then:

- \(\text{support\_max}_I = \max(0.88, 0.81, 0.22) = 0.88\)  
- \(\text{contradict\_max}_I = 0.10\)

Verdict for Islam = **supports** → 🟢 **Agrees**.

### 7.2 Christian side

Top retrieved passages:

- \(q_1\): Mark 15:37–39  
- \(q_2\): John 19:30–35  
- \(q_3\): Commentary affirming crucifixion

NLI outputs:

- \(q_1\): \(p_E = 0.05\), \(p_C = 0.84\), \(p_N = 0.11\) → contradicts  
- \(q_2\): \(p_E = 0.04\), \(p_C = 0.89\), \(p_N = 0.07\) → contradicts  
- \(q_3\): \(p_E = 0.07\), \(p_C = 0.78\), \(p_N = 0.15\) → contradicts

Then:

- \(\text{support\_max}_C = 0.07\)  
- \(\text{contradict\_max}_C = 0.89\)

Verdict for Christianity = **contradicts** → 🔴 **Disagrees**.

### 7.3 Combined verdict

- Islam: supports → 🟢 Agrees  
- Christianity: contradicts → 🔴 Disagrees  

Rule: one supports and the other contradicts → overall
**traditions_disagree** → 🔴 **Conflicted**.

The UI shows:

- Islam: 🟢 Agrees  
- Christianity: 🔴 Disagrees  
- Both sources: 🔴 Conflicted  

and you can inspect exactly which passages drove those scores.
"""
        )

    # 8. Limitations / can it be wrong?
    with st.expander("8. Can FACTR be wrong? What are its limitations?"):
        st.markdown(
            """
Yes. FACTR is a research prototype with several important limitations:

- It only sees the **curated KB** (Qur’an/Bible translations, chosen tafsīr/commentary).
  Any tradition, school or source that is missing will not be reflected.

- The **NLI model** can misinterpret verses, especially if they are highly metaphorical
  or context-dependent.

- Retrieval can miss key passages, or over-emphasise marginal ones.

- Verdicts depend on thresholds chosen for this prototype; different thresholds can
  produce slightly different “Agrees/Disagrees” splits.

For serious use, the system should **always** be treated as:

> “A way to quickly surface relevant passages and see how an AI model reads them,”  
> not as a final theological authority.

Your own expertise, and that of qualified scholars, should always have the final say.
"""
        )

    # 9. Does it replace scholars?
    with st.expander("9. Does FACTR replace scholars or fatwa councils?"):
        st.markdown(
            """
No. FACTR is **not** designed to issue fatwas, doctrinal rulings or pastoral advice.

It is a **tool for exploration and research**, intended to:

- highlight where a claim appears strongly supported or refuted by the texts in the KB,
- expose the **actual passages** being used,
- help spot misquotations, fabrications or selective readings.

Any serious doctrinal conclusion must be made by **human scholars** considering:

- the full breadth of sources,
- the language and context,
- the principles of their own legal and theological tradition.
"""
        )

    # 10. Bias, traditions and fairness
    with st.expander("10. How does FACTR handle different schools and potential bias?"):
        st.markdown(
            """
The current KB is intentionally **limited**:

- It focuses on widely used, broadly mainstream sources for each tradition.
- It does not yet include all schools (e.g. Shīʿa tafsīr, non-Evangelical Christian
  commentaries, other languages, etc.).

This means:

- Verdicts are **conditional** on the KB design.
- If one side has much richer coverage (e.g. more detailed tafsīr), that side can
  appear “stronger” simply because the KB is deeper.

The project explicitly treats this as a **known limitation** and a priority for
future work:

- broaden source diversity,
- document inclusion criteria,
- and allow users to see which sources are active in a given run.
"""
        )

    # 11. Privacy / what happens to my data?
    with st.expander("11. What happens to my debate data?"):
        st.markdown(
            """
This prototype works on **public or user-supplied debate data**:

- You can analyse YouTube debates that are already public.
- You can upload your own audio files for private analysis.

FACTR:

- does **not** modify or publish your files;
- stores intermediate artefacts (audio snapshots, transcripts, claims) locally
  on the server where it is run, for debugging and re-use;
- uses external AI models (e.g. OpenAI) *only* on the text snippets required
  for transcription, claim extraction and NLI.

In a production deployment, stronger controls (encryption, access control,
data retention policies) would be required. This prototype is meant for
research and internal testing.  
"""
        )
    # 12. Controls – max transcript chunks, max claims, top-K
    with st.expander("12. What do the 'Max transcript chunks', 'Max claims' and 'KB passages per claim' controls do?"):
        st.markdown(
            """
These three controls let you **trade off depth of analysis vs speed and API cost**.

### Max transcript chunks (0 = all)

- FACTR breaks a long transcript into **chunks** (segments) before extracting claims.
- This control limits **how many chunks** are used for claim extraction.

Rough intuition:

- `0`  → use **all** chunks (full transcript; slowest, most claims).  
- `1`  → only analyse the **first** chunk (e.g. first part of the debate).  
- `2`  → first two chunks, and so on.

**When to reduce it:**

- While testing the pipeline on a long YouTube debate, set this to `1` or `2`
  to get a quick preview of how FACTR behaves.
- For a final run on a key debate, set it back to `0` to cover the whole transcript.

---

### Max claims to verify (0 = all)

After extraction, FACTR can easily find **dozens of claims** in a long debate.
Each claim then triggers multiple AI calls (per tradition, per passage).

This control limits **how many claims** are actually passed to the verification step:

- `0`  → **all** extracted claims are verified.  
- `N`  → only the **first N** claims (in transcript order) are verified.

**Why this matters:**

- Verifying claims is the **most expensive** and **slowest** part (NLI + LLM calls).
- For quick testing, you can set this to something like `5` or `10`.
- For a serious analysis of a short debate, you can safely set it to `0` or a higher number.

---

### KB passages per claim (top-K)

This is labelled as:

> **“How many passages to retrieve from each tradition for each claim.”**

For each claim and each tradition (Islam / Christianity), FACTR:

1. finds the **top K** most relevant passages in the KB;  
2. runs NLI on each of those K passages;  
3. aggregates those signals into **Agrees / Disagrees / Divided / Insufficient**.

So if you set:

- `K = 5` → up to **5 Islamic** and **5 Christian** passages per claim.  
- `K = 10` → up to **10 Islamic** and **10 Christian** passages per claim.

**Trade-offs:**

- Larger K:
  - ✅ sees more context and “long tail” verses/commentary;  
  - ❌ more model calls → slower and more expensive;  
  - ❌ slightly higher risk of adding noisy or marginal passages.

- Smaller K:
  - ✅ faster and cheaper;  
  - ✅ focuses on the most central passages;  
  - ❌ might miss a less obvious but important verse.

A practical pattern is:

- during development: K ≈ **3–5**;  
- for a careful run on a small set of important claims: K ≈ **5–10**.

---

In summary:

- **Max transcript chunks** → *how much of the debate transcript to use*.  
- **Max claims** → *how many claims to send to verification*.  
- **KB passages per claim (top-K)** → *how much evidence to retrieve per claim per tradition*.

They are there to let you **control runtime and cost** without changing the underlying
scoring logic.
"""

        )
