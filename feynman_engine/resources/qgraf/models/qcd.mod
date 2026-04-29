% QCD model file for QGRAF
% Particles: uq/ua (u quark/antiquark), dq/da, sq/sa, cq/ca, bq/ba, tq/ta
%            G (gluon), gh/gha (ghost/antighost)

% Propagators: [field, dual-field, sign]
[uq, ua, -]
[dq, da, -]
[sq, sa, -]
[cq, ca, -]
[bq, ba, -]
[tq, ta, -]
[G, G, +]
[gh, gha, -]

% Vertices
[uq, ua, G]
[dq, da, G]
[sq, sa, G]
[cq, ca, G]
[bq, ba, G]
[tq, ta, G]
[G, G, G]
[G, G, G, G]
[gh, gha, G]
