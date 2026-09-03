# Python
import re
from typing import Tuple, Optional
from simstack.models.simple_table import SimpleTable

HC_EV_NM = 1239.8419843320026  # Planck*c in eV*nm

START_MARKER = "Excitation energies and oscillator strengths:"

_HEADER_RE = re.compile(
    r"""
    ^\s*Excited\ State\s+
    (?P<num>\d+)
    :\s*
    (?P<label>[^:\n]*?)               # Singlet-A etc. (kept but unused)
    \s+
    (?P<e_ev>[+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s*eV
    (?:\s+(?P<e_nm>[+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s*nm)?
    .*?
    f\s*=\s*(?P<f>[+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)
    .*?
    <S\*\*2>\s*=\s*(?P<s2>[+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)
    """,
    re.IGNORECASE | re.VERBOSE,
    )

_TRANSITION_RE = re.compile(
    r"""
    ^\s*
    (?P<o1>\d+)\s*->\s*(?P<o2>\d+)
    \s+
    (?P<coeff>[+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)
    (?:\s*[%].*)?                      # tolerate trailing comments like percentages
    $
    """,
    re.VERBOSE,
)

def _to_float(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    try:
        return float(s)
    except Exception:
        return None

def _ev_to_nm(e_ev: Optional[float]) -> Optional[float]:
    if e_ev is None or e_ev == 0:
        return None
    return HC_EV_NM / e_ev

def parse_gaussian_excited_states_file(path: str) -> Tuple[SimpleTable, SimpleTable]:
    """
    Parse a Gaussian TD-DFT excited-states section from a file into two SimpleTable instances.

    Behavior:
    - Starts parsing after the first line that matches START_MARKER.
    - Continues to EOF (no explicit end marker required).
    - Collects all 'Excited State ...' headers and subsequent transition lines anywhere after the marker.

    Returns:
        (states_table, transitions_table)
    """
    states_table = SimpleTable()
    states_table.add_column("number", "int")
    states_table.add_column("energy_ev", "float")
    states_table.add_column("energy_nm", "float")
    states_table.add_column("f", "float")
    states_table.add_column("s2", "float")

    transitions_table = SimpleTable()
    transitions_table.add_column("state", "int")
    transitions_table.add_column("orb1", "int")
    transitions_table.add_column("orb2", "int")
    transitions_table.add_column("coeff", "float")

    started = False
    current_state_num: Optional[int] = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            if not started:
                if START_MARKER in line:
                    started = True
                continue

            # Once started, scan the remainder of the file, collecting states and transitions
            m = _HEADER_RE.search(line)
            if m:
                current_state_num = int(m.group("num"))
                e_ev = _to_float(m.group("e_ev"))
                e_nm = _to_float(m.group("e_nm"))
                fval = _to_float(m.group("f"))
                s2 = _to_float(m.group("s2"))

                if e_nm is None:
                    e_nm = _ev_to_nm(e_ev)

                states_table.add_row(
                    {
                        "number": current_state_num,
                        "energy_ev": e_ev,
                        "energy_nm": e_nm,
                        "f": fval,
                        "s2": s2,
                    }
                )
                continue

            tm = _TRANSITION_RE.search(line)
            if tm and current_state_num is not None:
                o1 = int(tm.group("o1"))
                o2 = int(tm.group("o2"))
                coeff = _to_float(tm.group("coeff"))
                transitions_table.add_row(
                    {
                        "state": current_state_num,
                        "orb1": o1,
                        "orb2": o2,
                        "coeff": coeff,
                    }
                )

    return states_table, transitions_table

if __name__ == "__main__":
    states, transitions = parse_gaussian_excited_states_file('gaussian.log')
    print(states)
    print(transitions)