"""ground_keywords_strict.py  — SELF-CONTAINED, STRICT keyword grounding.

Loose (v4/full) counts a keyword grounded if it appears anywhere in the anchor_line
+/-2 paragraph window (thousands of chars, several sentences). STRICT narrows this to
the actual SENTENCE SPAN that the evidence quote aligns to in the body:
  - align evidence tokens to the body token stream (difflib longest contiguous match)
  - within the anchor paragraph, pick the minimal contiguous run of sentences that
    covers the evidence content-words -> matched_span_text (the cited sentence(s))
  - keyword_in_sentence_span = is the keyword literally in THAT span (left-boundary,
    flip detection unchanged). If the span can't be trusted (anchor_match<0.6) the row
    stays anchor_unreliable.

One row per (cif,axis,guest_key) evidence (1:1, dict guests split); keyword-less
evidence kept (keyword=none, sentence check only) -> same coverage as the full file.
Output keyword_grounding_strict.csv (UTF-8 BOM, ASCII labels, control chars stripped).
Existing files are never modified.
"""
import csv, os, re, sys, json, copy, unicodedata
from difflib import SequenceMatcher
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "papers_md"   # Marker-converted full texts (not redistributed; see examples/)
MP   = os.path.join(ROOT, "Main_process")
SRC  = os.path.join(HERE, "CR_Main_LLM_Extraction.csv")
OUT  = os.path.join(HERE, "keyword_grounding_strict.csv")
SUMM = os.path.join(HERE, "keyword_grounding_strict_summary.csv")
MIN_MATCH = 0.6
ENC  = "utf-8-sig"
GROUP_EN = {"pos": "positive", "neg": "negative", "rev": "reversibility", "dec": "decomposition"}

# ---------------- normalization ----------------
TAG=re.compile(r"<[^>]+>"); LATEX=re.compile(r"\\[a-zA-Z]+")
DASH=dict.fromkeys(map(ord,"‐‑‒–—―−"),"-"); DOT=dict.fromkeys(map(ord,"·•∙×")," ")
SYMS=re.compile(r"[${}^_~|*#]"); WORD=re.compile(r"[a-z0-9]+"); NONAL=re.compile(r"[^a-z0-9]+")
CTRL=re.compile(r"[\r\n\t\x00-\x1f]+")
SENT=re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9(“"])')
def norm(s):
    s=unicodedata.normalize("NFKC",s or ""); s=TAG.sub(" ",s); s=LATEX.sub(" ",s)
    s=s.translate(DASH).translate(DOT); s=SYMS.sub(" ",s); return s.lower()
def clean_field(s): return CTRL.sub(" ",(s or "")).strip()

# ---------------- CR_Main parsing ----------------
AXES={"solvent":"Evidence_sentence_solvent","temperature":"Evidence_sentence_temperature",
      "gas":"Evidence_sentence_gas adsorption/desorption"}
FLEXCOL={"solvent":"Flexibility_solvent","temperature":"Flexibility_temperature",
         "gas":"Flexibility_gas adsorption/desorption"}
NONE={"","none","nan","null","{}","[]","na","n/a"}
KV=re.compile(r'"\s*([^"]{1,80}?)\s*"\s*:\s*"(.*?)"\s*(?:,\s*"|\}\s*$|\}\s*,)',re.S)
def resolve_md(doi):
    if not doi: return None
    d=doi.strip().replace("/","@"); p=os.path.join(MP,d,d+".md")
    return p if os.path.exists(p) else None
def parse_evidence(cell):
    s=(cell or "").strip()
    if s.lower() in NONE: return []
    if s[:1] in "{[":
        try:
            obj=json.loads(s)
            if isinstance(obj,dict): return [(str(k),str(v)) for k,v in obj.items() if str(v).strip().lower() not in NONE]
            if isinstance(obj,list): return [("",str(v)) for v in obj if str(v).strip().lower() not in NONE]
        except Exception:
            pairs=KV.findall(s)
            if pairs: return [(k,v) for k,v in pairs if v.strip().lower() not in NONE]
            return [("",s.strip("{}"))]
    return [("",s)]

# ---------------- synonym dict (whole-md sentence coverage) ----------------
SYN_FAMILIES=[
 ["water","h2o","moisture","aqueous","hydration","hydrated","hydrate","dehydration","dehydrated","dehydrate","dehydrating","rehydration","rehydrated","rehydrate","dried","drying","desolvated","desolvation"],
 ["methanol","meoh","ch3oh","ch4o"],["ethanol","etoh","c2h5oh"],["dimethylformamide","dmf"],["dimethylacetamide","dma","dmac"],
 ["diethylformamide","def"],["acetonitrile","mecn","ch3cn","acn"],["dichloromethane","dcm","ch2cl2","methylene"],["chloroform","chcl3"],
 ["tetrahydrofuran","thf"],["dimethylsulfoxide","dmso","sulfoxide"],["isopropanol","iproh","isoproh","propan","isopropyl"],
 ["dimethyl","me2"],["nmp","pyrrolidone"],["acetone","propanone","me2co"],["carbon","ccl4"]]
SYN_INDEX={}
for fam in SYN_FAMILIES:
    fam=[m.replace("-","").replace(",","") for m in fam]
    for m in fam: SYN_INDEX.setdefault(m,set()).update(fam)
def _present(t,tokset,collapsed):
    if t in tokset: return True
    if any(c.isdigit() for c in t) and len(t)>2 and t in collapsed: return True
    fam=SYN_INDEX.get(t); return bool(fam and (fam & tokset))
def sentence_coverage(en,tokset,collapsed):
    t=[w for w in WORD.findall(en) if len(w)>2]
    if not t: return 0.0
    return sum(1 for w in t if _present(w,tokset,collapsed))/len(t)

# ---------------- keyword spec (v2 base + v3 aug) ----------------
def lb(*r): return [("lb",x) for x in r]
def sub(*p): return [("sub",x) for x in p]
KW_BASE={
 "flexible":("pos",lb("flexib")),"breathing":("pos",lb("breath")),
 "gate_opening":("pos",sub("gate-open","gate open")),"gate_closing":("pos",sub("gate-clos","gate clos")),
 "sc_sc":("pos",sub("sc-sc")),
 "single_crystal_to_single_crystal":("pos",sub("single-crystal-to-single","single crystal to single","single-crystalto-single")),
 "crystal_to_crystal":("pos",sub("crystal-to-crystal","crystal to crystal")),
 "phase_transition":("pos",sub("phase transit","phase-transit")),
 "structural_transition_transformation":("pos",sub("structural transit","structural transformat","structural change","structural rearrang")),
 "swelling":("pos",lb("swell")),"shrinkage":("pos",lb("shrink")),
 "expansion":("pos",lb("expansion","expand")),"contraction":("pos",lb("contract")),
 "narrow_pore":("pos",sub("narrow-pore","narrow pore","np-to-lp")),"large_pore":("pos",sub("large-pore","large pore","lp-to-np")),
 "open_closed_form":("pos",sub("open form","closed form","open-form","closed-form","open pore","closed pore","open-pore","closed-pore")),
 "guest_induced":("pos",sub("guest-induc","guest induc")),
 "adsorption_induced":("pos",sub("adsorption-induc","adsorption induc","adsorbate-induc","sorption-induc")),
 "stimuli_responsive":("pos",sub("stimuli-respons","stimuli respons","stimulus-respons")),
 "rearrangement":("pos",lb("rearrang")),"linker_rotation":("pos",sub("linker rotat","ligand rotat","linker-rotat")),
 "libration":("pos",lb("librat")),"tilting":("pos",lb("tilt")),
 "negative_thermal_expansion":("pos",sub("negative thermal expansion")),"NTE":("pos",[("word","nte")]),
 "rigid":("neg",lb("rigid")),"robust":("neg",lb("robust")),"unchanged":("neg",lb("unchang")),
 "retained":("neg",lb("retain")),"intact":("neg",lb("intact")),"maintained":("neg",lb("maintain")),
 "stable":("neg",lb("stable")),"stability":("neg",lb("stabilit")),
 "reversible":("rev",lb("revers")),"irreversible":("rev",lb("irrevers")),"recover":("rev",lb("recover")),
 "restore":("rev",lb("restor")),"regenerate":("rev",lb("regener")),"revert":("rev",lb("revert")),"recoverable":("rev",lb("recoverable")),
 "collapse":("dec",lb("collaps")),"amorphous":("dec",lb("amorph")),"amorphization":("dec",lb("amorphiz")),
 "decomposition":("dec",lb("decompos")),"degradation":("dec",lb("degrad")),"hysteresis":("dec",lb("hysteres")),"distortion":("dec",lb("distort"))}
AUG={
 "sc_sc":[("rx",r"(?<![a-z])sc[\s\-]sc(?![a-z])"),("rx",r"(?<![a-z])sc[\s\-]to[\s\-]sc(?![a-z])")],
 "single_crystal_to_single_crystal":[("rx",r"single[\s\-]?crystal[\s\-]?to[\s\-]?single[\s\-]?crystal")],
 "crystal_to_crystal":[("rx",r"(?<!single.)crystal[\s\-]?to[\s\-]?crystal")],
 "narrow_pore":[("word","np"),("rx",r"narrow[\s\-]?pore")],"large_pore":[("word","lp"),("rx",r"large[\s\-]?pore")],
 "phase_transition":[("sub","phase transformat"),("sub","phase-transformat")],
 "gate_opening":[("lb","gating"),("rx",r"gate[\s\-]?open")]}
KEYWORDS=copy.deepcopy(KW_BASE)
for kw,extra in AUG.items():
    g,pats=KEYWORDS[kw]; KEYWORDS[kw]=(g,pats+extra)
OPP={"reversible":"irreversible","irreversible":"reversible","rigid":"flexible","flexible":"rigid"}
def _mk(kind,pat):
    if kind=="lb":   r=re.compile(r"(?<![a-z])"+re.escape(pat));               return lambda t: r.search(t) is not None
    if kind=="word": r=re.compile(r"(?<![a-z])"+re.escape(pat)+r"(?![a-z])"); return lambda t: r.search(t) is not None
    if kind=="rx":   r=re.compile(pat);                                       return lambda t: r.search(t) is not None
    return lambda t: pat in t
MATCH={k:[_mk(kind,p) for kind,p in pats] for k,(g,pats) in KEYWORDS.items()}
def kw_in(t,kw): return any(m(t) for m in MATCH[kw])
def kws_present(t): return [k for k in KEYWORDS if kw_in(t,k)]

# ---------------- md loading + span alignment ----------------
def load_paper(path):
    raw=open(path,encoding="utf-8",errors="replace").read().splitlines()
    ntext=[norm(l) for l in raw]; ntok=[set(WORD.findall(t)) for t in ntext]
    toks,lines=[],[]
    for i,t in enumerate(ntext):
        for w in WORD.findall(t): toks.append(w); lines.append(i+1)
    tokset=set().union(*ntok) if ntok else set()
    collapsed=NONAL.sub(""," ".join(ntext))
    return dict(raw=raw,ntext=ntext,ntok=ntok,toks=toks,lines=lines,tokset=tokset,collapsed=collapsed)

def anchor_and_span(en, P):
    """returns (anchor_line, anchor_match_loose, span_raw). span = minimal contiguous
    run of sentences (in the anchor paragraph +/-1 line) covering the evidence words."""
    E=WORD.findall(en); need=set(t for t in E if len(t)>2)
    if not E or not P["toks"] or not need: return 0,0.0,""
    sm=SequenceMatcher(None,E,P["toks"],autojunk=False)
    m=sm.find_longest_match(0,len(E),0,len(P["toks"]))
    if m.size==0: return 0,0.0,""
    ln=P["lines"][m.b]
    a,b=max(0,ln-1-2),min(len(P["ntok"]),ln+2)
    win=set().union(*P["ntok"][a:b]) if b>a else set()
    amatch=len(need&win)/len(need)
    # sentences from anchor paragraph +/-1 line
    ra,rb=max(0,ln-1-1),min(len(P["raw"]),ln+1)
    region=" ".join(P["raw"][ra:rb]).strip()
    sents=[s for s in SENT.split(region) if s.strip()]
    if not sents: return ln,round(amatch,3),clean_field(P["raw"][ln-1] if 1<=ln<=len(P["raw"]) else "")
    stoks=[set(WORD.findall(norm(s))) for s in sents]
    covs=[len(st&need)/len(need) for st in stoks]
    i=covs.index(max(covs)); lo=hi=i; cur=set(stoks[i])
    def cc(): return len(cur&need)/len(need)
    improved=True
    while cc()<0.9 and improved:
        improved=False; best=None; bestgain=0.12
        for side,idx in ([("r",hi+1)] if hi+1<len(sents) else [])+([("l",lo-1)] if lo-1>=0 else []):
            gain=len((cur|stoks[idx])&need)/len(need)-cc()
            if gain>bestgain: bestgain=gain; best=(side,idx)
        if best:
            side,idx=best
            if side=="r": hi=idx
            else: lo=idx
            cur|=stoks[idx]; improved=True
    return ln,round(amatch,3)," ".join(sents[lo:hi+1]).strip()

def get_paper(cache,doi):
    if doi not in cache:
        p=resolve_md(doi); cache[doi]=load_paper(p) if p else None
    return cache[doi]

def wp_of(text_norm):
    return {"reversible":bool(re.search(r"(?<![a-z])revers",text_norm)),
            "irreversible":bool(re.search(r"(?<![a-z])irrevers",text_norm)),
            "rigid":bool(re.search(r"(?<![a-z])rigid",text_norm)),
            "flexible":bool(re.search(r"(?<![a-z])flexib",text_norm))}

# ---------------- run ----------------
def run():
    rows=list(csv.DictReader(open(SRC,encoding="utf-8-sig")))
    cache={}; out=[]; kwstat=[]; n_ev=0
    for r in rows:
        cif=r.get("cif_name") or r.get("﻿cif_name") or ""
        doi=(r.get("doi_clean") or r.get("DOI") or "").strip()
        P=get_paper(cache,doi)
        for axis,col in AXES.items():
            label=(r.get(FLEXCOL[axis]) or "").strip()
            for gk,sent in parse_evidence(r.get(col,"")):
                n_ev+=1; en=norm(sent)
                if P:
                    scov=round(sentence_coverage(en,P["tokset"],P["collapsed"]),3)
                    anchor,amatch,span_raw=anchor_and_span(en,P)
                else:
                    scov,anchor,amatch,span_raw=0.0,0,0.0,""
                sstatus="no_paper" if not P else ("grounded" if scov>=0.85 else "weak" if scov>=0.5 else "absent")
                reliable=(amatch>=MIN_MATCH) if P else False
                # loose window (+/-2 lines) + span, both normalized
                if P and anchor:
                    a,b=max(0,anchor-1-2),min(len(P["ntext"]),anchor+2)
                    loose=" ".join(P["ntext"][a:b]); loose_raw=clean_field(" ".join(P["raw"][a:b]))
                else:
                    loose,loose_raw="",""
                span_n=norm(span_raw); wp_span=wp_of(span_n); wp_loose=wp_of(loose)
                kws=kws_present(en) if P else []
                if kws:
                    in_span=[]; in_loose=[]; flips=[]
                    for k in kws:
                        isp=kw_in(span_n,k); ilo=kw_in(loose,k)
                        fl=(not isp and k in OPP and wp_span.get(OPP[k]) and not wp_span.get(k))
                        in_span.append(isp); in_loose.append(ilo); flips.append(bool(fl))
                        if reliable:
                            cited_nearby = (not isp) and ilo
                            kwstat.append((k,isp,ilo,bool(fl),cited_nearby))
                    groups=list(dict.fromkeys(GROUP_EN[KEYWORDS[k][0]] for k in kws))
                    keyword=";".join(kws); keyword_group=";".join(groups)
                    in_win_col=";".join("T" if x else "F" for x in in_loose)       # loose (+/-2)
                    in_span_col=";".join("T" if x else "F" for x in in_span)        # strict (sentence span)
                    flip_col=";".join("T" if x else "F" for x in flips)
                    if not reliable:        kstatus="anchor_unreliable"
                    elif all(in_span):      kstatus="grounded"
                    elif any(flips):        kstatus="flip"
                    elif any(in_span):      kstatus="partial"
                    else:                   kstatus="absent"
                else:
                    keyword=keyword_group=in_win_col=in_span_col=flip_col="none"; kstatus="no_keyword"
                out.append({"axis":axis,"cif":cif,"doi":doi,"guest_key":clean_field(gk),
                    "label":clean_field(label),"evidence":clean_field(sent),
                    "sentence_coverage":scov,"sentence_status":sstatus,
                    "keyword":keyword,"keyword_group":keyword_group,
                    "in_anchor_window":in_win_col,"keyword_in_sentence_span":in_span_col,
                    "anchor_line":anchor,"anchor_match":amatch,"anchor_reliable":("T" if reliable else "F"),
                    "flip_flag":flip_col,"keyword_status":kstatus,
                    "matched_span_text":clean_field(span_raw),"anchor_window_text":loose_raw})
    cols=["axis","cif","doi","guest_key","label","evidence","sentence_coverage","sentence_status",
          "keyword","keyword_group","in_anchor_window","keyword_in_sentence_span","anchor_line",
          "anchor_match","anchor_reliable","flip_flag","keyword_status","matched_span_text","anchor_window_text"]
    with open(OUT,"w",newline="",encoding=ENC) as f:
        w=csv.DictWriter(f,fieldnames=cols); w.writeheader()
        for o in out: w.writerow(o)

    # ---- comparison strict vs loose (per keyword occurrence, reliable rows) ----
    print(f"evidence units: {n_ev} | output rows: {len(out)}  (1:1 = {n_ev==len(out)})")
    tot=Counter(k for k,*_ in kwstat)
    loose_g=Counter(k for k,isp,ilo,fl,cn in kwstat if ilo)
    strict_g=Counter(k for k,isp,ilo,fl,cn in kwstat if isp)
    nearby=Counter(k for k,isp,ilo,fl,cn in kwstat if cn)
    print(f"\n{'keyword':36s}| reliable | loose% -> strict% | cited_nearby")
    srows=[]
    for kw,(g,_) in KEYWORDS.items():
        t=tot.get(kw,0)
        if t==0: continue
        lp=100*loose_g.get(kw,0)/t; sp=100*strict_g.get(kw,0)/t
        print(f"{kw:36s}| {t:6d}   | {lp:5.1f}% -> {sp:5.1f}% | {nearby.get(kw,0)}")
        srows.append([kw,GROUP_EN[g],t,loose_g.get(kw,0),strict_g.get(kw,0),nearby.get(kw,0),round(lp,1),round(sp,1)])
    with open(SUMM,"w",newline="",encoding=ENC) as f:
        w=csv.writer(f); w.writerow(["keyword","group","reliable_claims","loose_grounded","strict_grounded","cited_nearby","loose_pct","strict_pct"]); w.writerows(srows)
    T=len(kwstat); Lg=sum(1 for k,isp,ilo,fl,cn in kwstat if ilo); Sg=sum(1 for k,isp,ilo,fl,cn in kwstat if isp)
    N=sum(1 for k,isp,ilo,fl,cn in kwstat if cn)
    strict_absent=sum(1 for k,isp,ilo,fl,cn in kwstat if not isp)
    print(f"\nOVERALL (reliable keyword-claims={T}):")
    print(f"  loose grounded : {Lg}/{T} = {100*Lg/max(1,T):.1f}%")
    print(f"  strict grounded: {Sg}/{T} = {100*Sg/max(1,T):.1f}%   (drop {100*(Lg-Sg)/max(1,T):.1f} pts)")
    print(f"  strict-absent  : {strict_absent} | of which cited_nearby (kw in +/-2 paragraph, not the cited sentence): {N} ({100*N/max(1,strict_absent):.0f}%)")
    print("-> keyword_grounding_strict.csv / keyword_grounding_strict_summary.csv")

if __name__=="__main__":
    run()
