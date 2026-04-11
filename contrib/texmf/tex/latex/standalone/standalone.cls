% Minimal standalone.cls — accepts [border=Xpt] option, ignores it gracefully.
% Uses a fixed 20cm page; correct for TikZ-Feynman with pdf2svg.
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{standalone}[2024/01/01 v1.0 Minimal standalone class]
\DeclareOption*{}%  silently ignore all options (border=5pt, etc.)
\ProcessOptions\relax
\LoadClass{article}
\RequirePackage[paperwidth=20cm,paperheight=20cm,margin=5pt,noheadfoot]{geometry}
\pagestyle{empty}
\setlength{\parindent}{0pt}
\setlength{\topskip}{0pt}
