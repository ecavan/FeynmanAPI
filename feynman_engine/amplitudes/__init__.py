"""Amplitude backends and shared amplitude result types."""

from feynman_engine.amplitudes.types import AmplitudeResult
from feynman_engine.amplitudes.symbolic import (
    get_symbolic_amplitude,
    get_loop_integral_latex,
    get_tree_integral_latex,
)
from feynman_engine.amplitudes.loop import (
    get_loop_amplitude,
    get_loop_pv_decomposition,
    pv_reduce,
    LoopTopology,
    PVExpansion,
    A0Integral,
    B0Integral, B1Integral, B00Integral, B11Integral,
    C0Integral, C1Integral, C2Integral, C00Integral, C11Integral, C12Integral, C22Integral,
    D0Integral, D00Integral, D1Integral, D2Integral, D3Integral,
    D11Integral, D12Integral, D13Integral, D22Integral, D23Integral, D33Integral,
)
from feynman_engine.amplitudes.loop_tensor_reduction import auto_pv_reduce
from feynman_engine.amplitudes.looptools_bridge import (
    is_available as looptools_available,
    evaluate_pv_expansion,
)
from feynman_engine.amplitudes.analytic_integrals import (
    analytic_A0, analytic_B0, analytic_B1, analytic_B00,
    analytic_C0, analytic_D0,
    Delta_UV,
)
from feynman_engine.amplitudes.loop_curated import (
    get_loop_curated_results,
    evaluate_photon_selfenergy,
    evaluate_vertex_form_factor,
    evaluate_schwinger_amm,
    evaluate_vacuum_polarisation,
)
from feynman_engine.amplitudes.cross_section import (
    total_cross_section,
    total_cross_section_mc,
    total_cross_section_vegas,
    differential_cross_section,
)
from feynman_engine.amplitudes.nlo_cross_section import (
    nlo_cross_section,
    nlo_cross_section_qed,
    alpha_s_running,
    alpha_em_running,
)
from feynman_engine.amplitudes.phase_space import (
    rambo_massless,
    rambo_massive,
    total_cross_section_2to3,
    vegas_integrate,
    VegasGrid,
)
from feynman_engine.amplitudes.pdf import (
    PDFSet,
    LHAPDFSet,
    get_builtin_pdf,
    get_pdf,
    parton_luminosity,
)
from feynman_engine.amplitudes.hadronic import (
    hadronic_cross_section,
)
from feynman_engine.amplitudes.differential import (
    differential_distribution,
    hadronic_differential_distribution,
)

__all__ = [
    "AmplitudeResult",
    "get_symbolic_amplitude",
    "get_loop_integral_latex",
    "get_tree_integral_latex",
    "get_loop_amplitude",
    "get_loop_pv_decomposition",
    "pv_reduce",
    "auto_pv_reduce",
    "LoopTopology",
    "PVExpansion",
    # 1-point
    "A0Integral",
    # 2-point
    "B0Integral", "B1Integral", "B00Integral", "B11Integral",
    # 3-point
    "C0Integral", "C1Integral", "C2Integral",
    "C00Integral", "C11Integral", "C12Integral", "C22Integral",
    # 4-point
    "D0Integral", "D00Integral", "D1Integral", "D2Integral", "D3Integral",
    "D11Integral", "D12Integral", "D13Integral",
    "D22Integral", "D23Integral", "D33Integral",
    # analytic integrals
    "analytic_A0", "analytic_B0", "analytic_B1", "analytic_B00",
    "analytic_C0", "analytic_D0",
    "Delta_UV",
    "looptools_available",
    "evaluate_pv_expansion",
    "get_loop_curated_results",
    "evaluate_photon_selfenergy",
    "evaluate_vertex_form_factor",
    "evaluate_schwinger_amm",
    "evaluate_vacuum_polarisation",
    "total_cross_section",
    "total_cross_section_mc",
    "total_cross_section_vegas",
    "differential_cross_section",
    "rambo_massless",
    "rambo_massive",
    "total_cross_section_2to3",
    "vegas_integrate",
    "VegasGrid",
    "nlo_cross_section",
    "nlo_cross_section_qed",
    "alpha_s_running",
    "alpha_em_running",
    # PDFs and hadronic cross-sections
    "PDFSet",
    "LHAPDFSet",
    "get_builtin_pdf",
    "get_pdf",
    "parton_luminosity",
    "hadronic_cross_section",
    # Differential observables
    "differential_distribution",
    "hadronic_differential_distribution",
]
