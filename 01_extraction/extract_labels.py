# -*- coding: utf-8 -*-

import os
import json
import re
import time
import pandas as pd
from tqdm import tqdm
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from chromadb import PersistentClient

# ============================================================
# 1. Setting
# ============================================================
# The OpenAI API key is read from the environment; set it before running:
#   export OPENAI_API_KEY=...
if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("Set the OPENAI_API_KEY environment variable before running extract_labels.py")

CHROMA_PATH = "chroma_db"   # local Chroma store built from converted papers (not redistributed)
COLLECTION_NAME = "MOF_Paper"
MODEL_NAME = "gpt-5.2"
OUTPUT_FILE = "CR_Label.csv"
CHECKPOINT_EVERY = 10
CHECKPOINT_FILE = "CR_Label_Checkpoint.csv"

# DOIs for few-shot examples
EXAMPLE_DOIS = [
    "10.1021/acs.inorgchem.7b01818",  # Example 1: CFA-10
    "10.1021/jacs.7b03850",            # Example 2: UTSA-300
    "10.1021/jacs.7b05858",            # Example 3: Zn2(dobpdc)
]

# ============================================================
# 2. Initialization
# ============================================================
embedding_function = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

chroma_client = PersistentClient(path=CHROMA_PATH)
vectordb = Chroma(
    client=chroma_client,
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_function,
)

llm = ChatOpenAI(model=MODEL_NAME, temperature=0)

# ============================================================
# 3. Data Loading
# ============================================================
CR_file = pd.read_excel("../data/LLM_Extraction.xlsx")

# ============================================================
# 4. Functions
# ============================================================
def retrieve_by_doi(doi: str) -> str:
    results = vectordb._collection.get(ids=[doi])
    if not results["ids"]:
        return None
    # metadata가 없으므로 document만 반환
    return results["documents"][0]


def build_prompt(retrieved_text: str, cif_file: str, metal: str, doi: str,
                 example_texts: dict) -> str:
    """Build system prompt with Production Prompt + dynamic few-shot examples"""

    system_prompt = """OUTPUT FORMAT: Return a single JSON object only. No preamble, no markdown fences, no explanation before or after the JSON.

You are a materials scientist specializing in metal-organic frameworks (MOFs).

Your task is to read the provided paper text and determine whether the SPECIFIC MOF in the question is structurally flexible.

You must evaluate flexibility independently along exactly three stimulus axes:
1. gas adsorption/desorption
2. temperature
3. solvent

Return JSON only.

## Core Definition

A MOF is considered flexible only if an external stimulus applied to an already-synthesized MOF causes a framework structural transformation WITHOUT irreversible collapse, destruction, or one-way structural failure.

Framework transformations fall into these categories:
- Breathing
- Swelling
- Phase transition
- Linker rotation
- Subnetwork distortion and displacement
- Others: [brief description]

Use `Flexibility_type` ONLY when `Flexibility = "Yes"`.
Otherwise `Flexibility_type` must be `"none"`.

## Terminology

- `"none"` = this stimulus axis was not validly tested for this MOF
- `"N/A"` = this field does not apply to that stimulus axis
- `"Unknown"` = the stimulus axis was tested, but the flexibility outcome cannot be determined from the paper

## Priority Order (strict)

For each stimulus axis, apply these in order:
1. Exact MOF match
2. Valid experimental stimulus
3. Structural evidence outcome
4. Irreversibility override
5. Flexibility decision
6. Flexibility type assignment
7. Guest inducing / non-inducing consistency

## Step-by-Step Decision Procedure

For each stimulus axis:

### Step 1. Exact MOF match
Was THIS EXACT MOF experimentally tested under this axis?
- If NO -> output `"none"` for all fields for this axis.
- If YES -> continue.

Do NOT transfer evidence from another MOF in the same paper, even if it is isoreticular or similar.

### Step 2. Valid experimental stimulus
A stimulus is valid only if:
- it was applied to an already-synthesized MOF, AND
- the paper evaluates framework structural behavior under that stimulus using structural evidence or explicit framework-behavior interpretation.

Do NOT count the following as valid stimuli:
- synthesis-condition differences
- TGA alone
- thermal activation/degassing alone
- computational calculations alone
- non-structural property measurements alone
- precursor-to-product synthesis transformations

### Step 3. Structural evidence outcome
Classify the evidence as exactly one of these:

A. Explicit structural change observed
B. Explicitly no structural change observed
C. Structural outcome cannot be determined

Then:
- A -> continue to Step 4
- B -> `Flexibility = "No"`
- C -> `Flexibility = "Unknown"`

Important:
- XRD/PXRD intensity-only changes without peak-position shifts do NOT count as structural change.
- If solvent soaking/exchange is performed and PXRD/SCXRD confirms no structural change, this is a valid solvent stimulus with `Flexibility = "No"`.

### Step 4. Irreversibility override
If the paper describes the structural change with any destructive or irreversible wording, then `Flexibility = "No"`.

Examples include:
- irreversible
- structural collapse
- loss of crystallinity
- amorphization without recovery
- decomposition
- framework destruction
- crystal cracking
- could not be reversed
- did not recover the original structure
- failed to revert
- attempts to retransform failed

Important:
- sorption hysteresis is NOT evidence of irreversibility
- diminished performance over repeated cycles is NOT automatically irreversibility unless the framework is explicitly said to be permanently altered

### Step 5. Flexibility decision
If destructive/irreversible wording is absent:

Assign `Flexibility = "Yes"` when:
- reversibility is explicitly demonstrated, OR
- the paper reports characteristic flexible behavior such as gate-opening, breathing, or stepwise/hysteretic sorption attributable to framework transformation, without destructive wording

Assign `Flexibility = "Unknown"` when:
- structural change is mentioned but framework involvement or reversibility remains ambiguous

### CRITICAL: Step-shaped isotherms from appended-group chemistry

Step-shaped or hysteretic isotherms caused by CHEMICAL REACTIONS on appended functional groups (for example, amine-CO2 chemisorption forming ammonium carbamate species) are NOT framework flexibility, even though they may superficially resemble gate-opening or breathing.

If the paper attributes unusual uptake to reactions on appended groups while describing the framework as rigid, scaffold-like, or relatively unchanged:
- `Flexibility = "Unknown"` if framework involvement remains ambiguous
- `Flexibility = "No"` if the paper explicitly rules out framework deformation
- `Flexibility` is NOT `"Yes"` unless the paper provides direct structural evidence of framework deformation

## Stimulus-Specific Guidance

### Temperature
Valid:
- VT-XRD / VT-PXRD showing lattice parameter change, NTE/PTE, anisotropic thermal expansion (ATE), or phase transition
- temperature-driven reversible or irreversible SC-SC transformation of an already-synthesized MOF
- Anomalous or large-magnitude thermal expansion (e.g., colossal NTE/PTE, ATE with coefficients far exceeding normal inorganic solids) constitutes framework flexibility under temperature stimulus. If VT-XRD/VT-PXRD demonstrates such behavior, report Flexibility = "Yes" and Flexibility_type = the appropriate expansion type (e.g., "Anisotropic thermal expansion", "Negative thermal expansion").

Invalid:
- TGA (thermogravimetric analysis) that only reports mass loss, decomposition temperature, or framework collapse. TGA alone measures thermal STABILITY, not structural FLEXIBILITY. Do NOT list "temperature" as a stimulus based solely on TGA data unless VT-XRD or other structural characterization confirms a structural transformation BEFORE decomposition.
- Thermal activation/degassing (e.g., heating under vacuum to remove solvent) is a sample preparation step, NOT a temperature stimulus, unless the paper explicitly reports a structural phase transition during this process with crystallographic evidence. Even if PXRD is measured before and after activation to confirm stability, this is NOT a valid temperature stimulus ? it is a stability check for sample preparation.
- Precursor-to-product synthesis transformation

### Solvent
Valid:
- soaking / immersion / solvent exchange of an already-synthesized MOF with structural characterization
- includes cases where no structural change occurs -> `Flexibility = "No"`

Invalid:
- different solvents used only during synthesis to obtain different products
- Stability/humidity tests where the MOF is exposed to solvents or moisture and only PXRD stability confirmation or surface area reduction is reported, without any discussion of structural transformation. If no structural change occurs and no structural ambiguity exists, this is a stability check, NOT a flexibility-relevant stimulus. Report "none".
- Exposing the MOF to solvents for non-structural measurements only (e.g., luminescence sensing, dye adsorption, catalysis) where PXRD is measured only to confirm framework retention. If the paper's purpose is to show the MOF survives the solvent rather than to investigate structural response, this is NOT a valid solvent stimulus unless a structural change is actually observed or discussed.
 
Important distinction:
- "Soaking/immersion with structural characterization showing NO structural change" -> valid solvent stimulus with Flexibility = "No". This applies when the paper INTENTIONALLY tests the MOF's structural response to solvent exchange.
- "Stability test or sensing experiment where PXRD confirms framework retention" -> NOT a valid solvent stimulus. Report "none". This applies when the paper's goal is to demonstrate robustness, not to investigate flexibility.

### Gas adsorption/desorption
Valid:
- sorption isotherms with hysteresis, stepwise uptake, gate-opening, or breathing-like behavior
- sorption experiments explicitly discussed in terms of framework response
- sorption + structural characterization confirming or ruling out deformation
- ordinary sorption behavior if the paper explicitly uses it to argue framework rigidity or absence of response

Invalid:
- zero/negligible uptake with no structural interpretation
- routine adsorption reported with no relevance to framework behavior

Important:
- reversible adsorption/desorption of gas molecules is NOT the same as reversible framework transformation
- step-shaped isotherms caused by chemistry on appended functional groups rather than framework deformation may be `"No"` or `"Unknown"`, depending on the paper

## Guest-Specific Rule

If different guest molecules under the same stimulus produce different flexibility outcomes, use nested dictionaries for:
- `Flexibility`
- `Flexibility_type`
- `Evidence_sentence`

Guest labels may include condition qualifiers such as temperature when the paper distinguishes outcomes by condition.

`Guest_inducing_flexibility` must include ONLY guests with `Flexibility = "Yes"`.
`Guest_non_inducing_flexibility` must include guests with `Flexibility = "No"` or `"Unknown"`.

## Missing Information Rule

If a required value is absent from the paper, output `"Unknown"`.
Never invent names, pressures, temperatures, figure names, chemical formulas, or citations.

## Output Format

Return JSON only in exactly this structure:

```json
{
  "MOF": "...",
  "Node": "...",
  "Ligand": "...",
  "Stimulus": "[solvent; temperature; gas adsorption/desorption]",
  "Guest_inducing_flexibility": {
    "solvent": "...",
    "temperature": "none",
    "gas adsorption/desorption": "..."
  },
  "Guest_non_inducing_flexibility": {
    "solvent": "...",
    "temperature": "none",
    "gas adsorption/desorption": "..."
  },
  "Temperature": {
    "solvent": "none",
    "temperature": "...",
    "gas adsorption/desorption": "..."
  },
  "Pressure": {
    "solvent": "N/A",
    "temperature": "N/A",
    "gas adsorption/desorption": "..."
  },
  "Flexibility": {
    "solvent": "...",
    "temperature": "...",
    "gas adsorption/desorption": "..."
  },
  "Flexibility_type": {
    "solvent": "...",
    "temperature": "...",
    "gas adsorption/desorption": "..."
  },
  "Evidence_sentence": {
    "solvent": "...",
    "temperature": "...",
    "gas adsorption/desorption": "..."
  },
  "Isotherm_figure_name": {
    "solvent": "none",
    "temperature": "none",
    "gas adsorption/desorption": "..."
  },
  "Confidence": "high/medium/low",
  "Reason": "..."
}
```

If different guests under the same axis give different outcomes, use nested dictionaries for that axis, for example:

```json
"Flexibility": {
  "gas adsorption/desorption": {
    "C2H2": "Yes",
    "CO2 (195 K)": "Yes",
    "CO2 (273 K)": "No"
  }
}
```

## Final Self-Check Before Output (MANDATORY)

You MUST walk through ALL of the following checks on your draft answer. If ANY check fails, FIX the answer before outputting.

CHECK 1 (Guest <-> Flexibility alignment - MOST COMMON ERROR):
  This is the single most frequently failed check. Execute with extreme care.
  For EACH stimulus axis, scan Guest_inducing_flexibility:
  - If Guest_inducing lists ANY guest, verify: does that EXACT guest have
    Flexibility = "Yes" (not "No", not "Unknown")?
  - If Flexibility = "No" for that guest: MOVE the guest to Guest_non_inducing.
    Set Guest_inducing = "none" for that axis. No exceptions.
  - If Flexibility = "Unknown" for that guest: Set Guest_inducing = "none".
  - COMMON MISTAKE: A guest that causes irreversible structural change has
    Flexibility = "No". Such a guest MUST NOT appear in Guest_inducing.
    "Inducing" refers to inducing FLEXIBILITY (reversible transformation),
    not inducing any structural change.

CHECK 2 (Flexibility_type alignment):
  For EACH stimulus axis:
  - If Flexibility = "No" or "Unknown" -> Flexibility_type MUST be "none".
  - If Flexibility = "Yes" -> Flexibility_type MUST list at least one type.

CHECK 3 (MOF identity):
  For EACH stimulus you listed:
  - Was the experiment performed on THIS EXACT MOF (the one matching the CIF name)?
  - If the experiment was performed on a DIFFERENT compound in the paper -> remove that stimulus entirely.

CHECK 4 (Irreversibility):
  For EACH stimulus with Flexibility = "Yes":
  - Does the paper describe the structural change as "irreversible"? -> Change to "No".
  - Does the paper report "structural collapse", "loss of crystallinity", or "decomposition"? -> Change to "No".

CHECK 5 (Stimulus validity):
  For EACH stimulus you listed:
  - Is it based on a real experiment (not computation, not synthesis conditions)?
  - Was structural characterization performed (XRD, PXRD, SCXRD)?
  - If either answer is no -> remove that stimulus.

CHECK 6 (Stimulus-Flexibility consistency):
  - Any axis with Flexibility = "none" must NOT appear in Stimulus.
  - If a valid tested stimulus explicitly shows no structural change, use "No" not "none".

CHECK 7 (JSON validity):
  - Return valid JSON only. No preamble, no markdown fences.

## Examples

### Example 1: CFA-10 (temperature=No, gas=No)

<PAPER>
""" + example_texts.get(EXAMPLE_DOIS[0], "[Example paper text not available]") + """
</PAPER>

```json
{
  "MOF": "[Fe(BPZ)] (CFA-10)",
  "Node": "(Fe(II) node) [Fe]",
  "Ligand": "(BPZ) [4,4'-bipyrazolate]",
  "Stimulus": "[temperature; gas adsorption/desorption]",
  "Guest_inducing_flexibility": {
    "solvent": "none",
    "temperature": "none",
    "gas adsorption/desorption": "none"
  },
  "Guest_non_inducing_flexibility": {
    "solvent": "none",
    "temperature": "none",
    "gas adsorption/desorption": "Ar; O2"
  },
  "Temperature": {
    "solvent": "none",
    "temperature": "573 K",
    "gas adsorption/desorption": "[183 K; 193 K; 203 K; 213 K; 87 K]"
  },
  "Pressure": {
    "solvent": "N/A",
    "temperature": "N/A",
    "gas adsorption/desorption": "up to 1 bar"
  },
  "Flexibility": {
    "solvent": "none",
    "temperature": "No",
    "gas adsorption/desorption": "No"
  },
  "Flexibility_type": {
    "solvent": "none",
    "temperature": "none",
    "gas adsorption/desorption": "none"
  },
  "Evidence_sentence": {
    "solvent": "none",
    "temperature": "CFA-10 was synthesized by an irreversible thermal phase transformation. All attempts to retransform CFA-10 to CFA-10-as were not successful, due to the loss of crystallinity of all investigated samples.",
    "gas adsorption/desorption": "Thus, in the first measurement, the sample was clearly oxidized at 183 K even at a low partial pressure of oxygen. After this measurement, the sample was evacuated, and the isotherm was recorded under the same conditions, revealing a Langmuir-type oxygen sorption isotherm."
  },
  "Isotherm_figure_name": {
    "solvent": "none",
    "temperature": "none",
    "gas adsorption/desorption": "_page_4_Figure_5.jpeg"
  },
  "Confidence": "high",
  "Reason": "Temperature: The SC-to-SC transformation from CFA-10-as to CFA-10 is explicitly irreversible with loss of crystallinity. Flexibility = No. Gas adsorption/desorption: The stepwise O2 uptake at 183 K is due to irreversible Fe(II)->Fe(III) oxidation, not framework flexibility. Ar shows standard Type I behavior. Flexibility = No."
}
```

### Example 2: UTSA-300 (guest-resolved gas adsorption)

<PAPER>
""" + example_texts.get(EXAMPLE_DOIS[1], "[Example paper text not available]") + """
</PAPER>

```json
{
  "MOF": "[Zn(dps)2(SiF6)] (UTSA-300)",
  "Node": "(Zn(II) node) [Zn2+]",
  "Ligand": "(dps) [4,4'-dipyridylsulfide]",
  "Stimulus": "[gas adsorption/desorption]",
  "Guest_inducing_flexibility": {
    "solvent": "none",
    "temperature": "none",
    "gas adsorption/desorption": "C2H2; CO2 (195 K)"
  },
  "Guest_non_inducing_flexibility": {
    "solvent": "none",
    "temperature": "none",
    "gas adsorption/desorption": "CO2 (273-298 K); C2H4"
  },
  "Temperature": {
    "solvent": "none",
    "temperature": "none",
    "gas adsorption/desorption": "[195 K; 273 K; 298 K]"
  },
  "Pressure": {
    "solvent": "N/A",
    "temperature": "N/A",
    "gas adsorption/desorption": "1 bar"
  },
  "Flexibility": {
    "solvent": "none",
    "temperature": "none",
    "gas adsorption/desorption": {
      "C2H2": "Yes",
      "CO2 (195 K)": "Yes",
      "CO2 (273-298 K)": "No",
      "C2H4": "No"
    }
  },
  "Flexibility_type": {
    "solvent": "none",
    "temperature": "none",
    "gas adsorption/desorption": {
      "C2H2": "Phase transition; Linker rotation",
      "CO2 (195 K)": "Phase transition",
      "CO2 (273-298 K)": "none",
      "C2H4": "none"
    }
  },
  "Evidence_sentence": {
    "solvent": "none",
    "temperature": "none",
    "gas adsorption/desorption": {
      "C2H2": "The loading of the C2D2 molecule broke the original hydrogen bond inside the closed-pore UTSA-300a, accompanied by the inversion of distorted ligand conformation, resulting in a fully expanded framework.",
      "CO2 (195 K)": "The CO2 sorption isotherms exhibit stepwise type with a hysteresis, suggesting UTSA-300a can be expanded to open-pore state.",
      "CO2 (273-298 K)": "The optimal sorption site for CO2 cannot break the original hydrogen bond under ambient conditions to open the closed-pore structure.",
      "C2H4": "The insufficient C2H4 adsorption can be attributed to complete size exclusion rather than structural flexibility."
    }
  },
  "Isotherm_figure_name": {
    "solvent": "none",
    "temperature": "none",
    "gas adsorption/desorption": "_page_3_Figure_5.jpeg"
  },
  "Confidence": "high",
  "Reason": "C2H2 induces closed-to-open pore transition with ligand conformational change. CO2 at 195 K shows stepwise uptake with hysteresis. At ambient temperature CO2 cannot trigger the transition. C2H4 excluded by size."
}
```

### Example 3: Zn2(dobpdc) (Unknown due to appended-group chemistry)

<PAPER>
""" + example_texts.get(EXAMPLE_DOIS[2], "[Example paper text not available]") + """
</PAPER>

```json
{
  "MOF": "Zn2(dobpdc)",
  "Node": "(Zn(II) node) [Zn2]",
  "Ligand": "(dobpdc) [4,4'-dioxidobiphenyl-3,3'-dicarboxylate]",
  "Stimulus": "[gas adsorption/desorption]",
  "Guest_inducing_flexibility": {
    "solvent": "none",
    "temperature": "none",
    "gas adsorption/desorption": "none"
  },
  "Guest_non_inducing_flexibility": {
    "solvent": "none",
    "temperature": "none",
    "gas adsorption/desorption": "CO2"
  },
  "Temperature": {
    "solvent": "none",
    "temperature": "none",
    "gas adsorption/desorption": "[298 K; 313 K; 323 K; 348 K; 373 K; 393 K]"
  },
  "Pressure": {
    "solvent": "N/A",
    "temperature": "N/A",
    "gas adsorption/desorption": "up to 1.2 bar"
  },
  "Flexibility": {
    "solvent": "none",
    "temperature": "none",
    "gas adsorption/desorption": "Unknown"
  },
  "Flexibility_type": {
    "solvent": "none",
    "temperature": "none",
    "gas adsorption/desorption": "none"
  },
  "Evidence_sentence": {
    "solvent": "none",
    "temperature": "none",
    "gas adsorption/desorption": "Step-shaped CO2 isotherms arise from cooperative CO2 insertion into metal-amine bonds. The framework is described as a relatively rigid crystalline scaffold."
  },
  "Isotherm_figure_name": {
    "solvent": "none",
    "temperature": "none",
    "gas adsorption/desorption": "none"
  },
  "Confidence": "medium",
  "Reason": "Step-shaped CO2 isotherms observed, but the paper attributes this to amine-CO2 chemisorption on appended groups, not framework deformation. The framework is called relatively rigid. Unknown because framework involvement is ambiguous."
}
```
"""

    user_message = f"""<QUERY>
<PAPER>
{retrieved_text}
</PAPER>

MOF corresponding to cif file name {cif_file} which the constituent metal node is {metal} in the doi {doi} shows flexibility? Answer like the examples above. The CIF file name may use IUPAC/CCDC systematic naming while the paper refers to the same MOF by a short name. Match the CIF to the correct MOF by comparing metal, linker, and structural description. Extract names as they appear in paper text.
</QUERY>"""

    return system_prompt, user_message


def _parse_llm_json(raw: str) -> dict:
    """
    Robust JSON parser for LLM output.
    Handles: markdown wrappers, unicode issues, trailing commas,
    control characters, single-quote dicts, truncated output, etc.
    Raises json.JSONDecodeError if all recovery attempts fail.
    """
    text = raw.strip()

    # -- Step 1: Strip markdown code fences --
    text = re.sub(r'^```(?:json)?\s*\n?', '', text)
    text = re.sub(r'\n?```\s*$', '', text)

    # -- Step 2: Extract outermost JSON object { ... } --
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        text = match.group(0)

    # -- Step 3: Remove BOM --
    text = text.lstrip('\ufeff')

    # -- Step 4: Replace unicode characters that are safe to replace always --
    replacements = {
        '\u2013': '-', '\u2014': '-', '\u2015': '-',
        '\u2018': "'", '\u2019': "'",
        '\u2192': '->', '\u2190': '<-', '\u2194': '<->',
        '\u2026': '...', '\u00b0': ' deg', '\u2103': ' degC',
        '\u00d7': 'x', '\u00b1': '+/-',
        '\u2082': '2', '\u2083': '3', '\u2084': '4',
        '\u03b1': 'alpha', '\u03b2': 'beta', '\u03b3': 'gamma',
        '\u00c5': 'A',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # -- Step 5: Remove control characters (except \n \r \t) --
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', text)

    # -- Step 6: Fix newlines inside JSON string values --
    def fix_string_newlines(m):
        s = m.group(0)
        inner = s[1:-1]
        inner = inner.replace('\n', '\\n').replace('\r', '\\r')
        return '"' + inner + '"'
    text = re.sub(r'"(?:[^"\\]|\\.)*"', fix_string_newlines, text)

    # -- Step 7: Fix trailing commas --
    text = re.sub(r',\s*([}\]])', r'\1', text)

    # -- Step 8: Remove JS-style comments --
    text = re.sub(r'(?<!["\w])//[^\n]*', '', text)

    # -- Step 9: Try parsing as-is --
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # -- Step 10: Handle curly double quotes -> single quotes --
    text_fixed = text.replace('\u201c', "'").replace('\u201d', "'")
    try:
        return json.loads(text_fixed)
    except json.JSONDecodeError:
        pass

    # -- Step 11: Single-quoted Python-style dicts --
    text_sq = text_fixed.replace("'", '"')
    try:
        return json.loads(text_sq)
    except json.JSONDecodeError:
        pass

    # -- Step 12: Fix unescaped backslashes --
    text_bs = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text_fixed)
    try:
        return json.loads(text_bs)
    except json.JSONDecodeError:
        pass

    # -- Step 13: Handle truncated JSON (close open brackets) --
    truncated = text_fixed.rstrip()
    open_braces = truncated.count('{') - truncated.count('}')
    open_brackets = truncated.count('[') - truncated.count(']')
    if open_braces > 0 or open_brackets > 0:
        if truncated[-1] not in ['"', '}', ']', ',']:
            truncated += '"'
        truncated += ']' * max(open_brackets, 0)
        truncated += '}' * max(open_braces, 0)
        try:
            return json.loads(truncated)
        except json.JSONDecodeError:
            pass

    # -- Step 14: ast.literal_eval as last resort --
    try:
        import ast
        result = ast.literal_eval(text_fixed)
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError):
        pass

    # All attempts failed - raise for retry logic
    return json.loads(text_fixed)


def extract_flexibility(llm, retrieved_text: str, cif_file: str, metal: str, doi: str,
                        example_texts: dict, max_retries: int = 2) -> dict:
    """Send prompt to LLM and parse JSON result"""

    system_prompt, user_message = build_prompt(retrieved_text, cif_file, metal, doi, example_texts)

    for attempt in range(max_retries + 1):
        try:
            response = llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ])

            answer = response.content.strip()
            # Print
            print(f"[system_prompt length: {len(system_prompt)} chars]")
            print('-' * 50)
            print(user_message)  # 논문 텍스트 전문
            print('-' * 50)
            print(answer)

            record = _parse_llm_json(answer)
            return record

        except json.JSONDecodeError as e:
            if attempt < max_retries:
                print(f"  JSON parse failed (attempt {attempt+1}), retrying...")
                continue
            print(f"  JSON parse final failure: {e}")
            return None

        except Exception as e:
            print(f"  API error: {e}")
            if attempt < max_retries:
                time.sleep(5)
                continue
            return None


# ============================================================
# 5. Pre-fetch example texts from ChromaDB
# ============================================================
print("Loading few-shot example texts from ChromaDB...")
example_texts = {}
for ex_doi in EXAMPLE_DOIS:
    text = retrieve_by_doi(ex_doi)
    if text:
        example_texts[ex_doi] = text
        print(f"  Loaded example: {ex_doi} ({len(text)} chars)")
    else:
        print(f"  WARNING: Example DOI {ex_doi} not found in ChromaDB")
        example_texts[ex_doi] = "[Paper text not available in database]"

# ============================================================
# 6. Flatten nested JSON to flat columns
# ============================================================
# 3 stimulus axes only (no pressure axis)
STIMULUS_KEYS = ["solvent", "temperature", "gas adsorption/desorption"]

# Updated: Evidence_sentence replaces Sentence_for_stimulus + Sentence_for_flexibility
NESTED_FIELDS = [
    "Guest_inducing_flexibility", "Guest_non_inducing_flexibility",
    "Temperature", "Pressure", "Flexibility", "Flexibility_type",
    "Evidence_sentence",
    "Isotherm_figure_name",
]
# Fields that are NOT nested (single value)
FLAT_FIELDS = ["MOF", "Node", "Ligand", "Stimulus", "Confidence", "Reason"]

# Build flat column list
columns = list(FLAT_FIELDS)
for field in NESTED_FIELDS:
    for stim in STIMULUS_KEYS:
        columns.append(f"{field}_{stim}")


def flatten_record(record: dict) -> dict:
    """Flatten nested stimulus-keyed JSON into flat column dict"""
    flat = {}

    # Copy flat fields directly
    for field in FLAT_FIELDS:
        flat[field] = record.get(field, "")

    # Expand nested fields
    for field in NESTED_FIELDS:
        value = record.get(field, {})
        if isinstance(value, dict):
            for stim in STIMULUS_KEYS:
                cell = value.get(stim, "none")
                # If sub-value is also a dict (guest-resolved), convert to string
                if isinstance(cell, dict):
                    cell = json.dumps(cell, ensure_ascii=False)
                flat[f"{field}_{stim}"] = cell
        else:
            # If LLM returned a single value instead of dict, put it in all stimulus columns
            for stim in STIMULUS_KEYS:
                flat[f"{field}_{stim}"] = value

    return flat


# ============================================================
# 7. Main loop
# ============================================================

answer_list = []

for i in tqdm(range(len(CR_file)), desc="Processing MOFs"):
    doi = CR_file["DOI"].values[i]
    metal = CR_file["metal"].values[i]
    cif_file = CR_file['IPUAC'][i]

    # Step 1: Retrieve text by DOI (try original, upper, lower)
    retrieved_text = None
    for doi_variant in [doi, doi.upper(), doi.lower()]:
        retrieved_text = retrieve_by_doi(doi_variant)
        if retrieved_text is not None:
            break

    if retrieved_text is None:
        print(f"[{i}] DOI {doi} -- no data found (tried original/upper/lower), skipping")
        answer_list.append({col: "" for col in columns})
        continue

    # Step 2: Extract flexibility via LLM
    record = extract_flexibility(llm, retrieved_text, cif_file, metal, doi, example_texts)
    if record:
        clean_record = flatten_record(record)
    else:
        clean_record = {col: "" for col in columns}
    answer_list.append(clean_record)

    # Checkpoint save
    if (i + 1) % CHECKPOINT_EVERY == 0:
        df_temp = pd.DataFrame(answer_list)
        temp_df = pd.concat([CR_file.iloc[: len(answer_list)], df_temp], axis=1)
        temp_df.to_csv(CHECKPOINT_FILE, index=False, encoding="utf-8-sig")
        print(f"  Checkpoint saved: {len(answer_list)} completed")

# ============================================================
# 8. Post-processing validation + Final save
# ============================================================

# Add postprocess check columns
PP_COLUMNS = ["pp_auto_fixed", "pp_manual_review", "pp_flags"]

try:
    from postprocess import validate_and_fix, batch_summary

    print("\n" + "=" * 60)
    print("Running post-processing validation (v2)...")
    print("=" * 60)

    validation_results = []
    for i, row in enumerate(answer_list):
        # Reconstruct nested dict from flat columns for validation
        record = {}
        for f in FLAT_FIELDS:
            record[f] = row.get(f, "")
        for f in NESTED_FIELDS:
            record[f] = {}
            for stim in STIMULUS_KEYS:
                val = row.get(f"{f}_{stim}", "none")
                if isinstance(val, str) and val.startswith('{'):
                    try:
                        val = json.loads(val)
                    except json.JSONDecodeError:
                        pass
                record[f][stim] = val

        # Retrieve paper text for CHECK_4 irreversibility scan
        doi = CR_file["DOI"].values[i] if i < len(CR_file) else ""
        paper_text = retrieve_by_doi(doi) if doi else ""
        if paper_text is None:
            paper_text = ""

        result = validate_and_fix(record, paper_text=paper_text)
        validation_results.append(result)

        # Print issues
        if not result.is_clean:
            cif_name = CR_file['cif_name'].values[i] if i < len(CR_file) else '?'
            print(f"\n--- Item {i} ({cif_name}) ---")
            print(result.summary())

        # --- Append postprocess columns to answer_list ---
        auto_checks = sorted(set(iss.check for iss in result.auto_fixed))
        manual_checks = sorted(set(iss.check for iss in result.manual_review))

        # Build detailed flag string: "CHECK_1(auto); CHECK_2(review); CHECK_4(review)"
        flags = []
        for iss in result.issues:
            tag = "auto" if iss.auto_fixed else "review"
            flags.append(f"{iss.check}({tag})")
        # Deduplicate while preserving order
        seen = set()
        unique_flags = []
        for f in flags:
            if f not in seen:
                seen.add(f)
                unique_flags.append(f)

        row["pp_auto_fixed"] = "; ".join(auto_checks) if auto_checks else ""
        row["pp_manual_review"] = "; ".join(manual_checks) if manual_checks else ""
        row["pp_flags"] = "; ".join(unique_flags) if unique_flags else ""

        # Also update the flattened row with auto-fixed values from postprocess
        fixed_record = result.fixed
        fixed_flat = flatten_record(fixed_record)
        for key, val in fixed_flat.items():
            row[key] = val

    print("\n" + batch_summary(validation_results))

except ImportError:
    print("\npostprocess.py (v2) not found - skipping validation.")
    print("Place postprocess.py (v2) in the same directory to enable.")
    for row in answer_list:
        row["pp_auto_fixed"] = ""
        row["pp_manual_review"] = ""
        row["pp_flags"] = ""

# Build final column list (original columns + postprocess columns)
all_columns = columns + PP_COLUMNS

# Final save
df_answers = pd.DataFrame(answer_list)
final_df = pd.concat([CR_file, df_answers[all_columns]], axis=1)
final_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"\nDone! {OUTPUT_FILE} saved ({len(final_df)} records)")

# Summary stats
if 'validation_results' in dir():
    n_clean = sum(1 for r in validation_results if r.is_clean)
    n_auto = sum(1 for r in validation_results if r.auto_fixed and not r.manual_review)
    n_manual = sum(1 for r in validation_results if r.manual_review)
    print(f"\n  Clean (no issues):     {n_clean}/{len(validation_results)}")
    print(f"  Auto-fixed only:       {n_auto}/{len(validation_results)}")
    print(f"  Needs manual review:   {n_manual}/{len(validation_results)}")