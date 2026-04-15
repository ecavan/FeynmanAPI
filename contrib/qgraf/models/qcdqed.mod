% QCD+QED model file for QGRAF
% QCD model extended with photon for mixed processes
% (e.g. u u~ -> gamma g, u g -> u gamma)

% Propagators: [field, dual-field, sign]
[uq, ua, -]
[dq, da, -]
[sq, sa, -]
[cq, ca, -]
[bq, ba, -]
[tq, ta, -]
[G, G, +]
[gh, gha, -]
[A, A, +]

% Vertices
% quark-gluon
[uq, ua, G]
[dq, da, G]
[sq, sa, G]
[cq, ca, G]
[bq, ba, G]
[tq, ta, G]
% triple and quartic gluon
[G, G, G]
[G, G, G, G]
% ghost-gluon
[gh, gha, G]
% quark-photon
[uq, ua, A]
[dq, da, A]
[sq, sa, A]
[cq, ca, A]
[bq, ba, A]
[tq, ta, A]
