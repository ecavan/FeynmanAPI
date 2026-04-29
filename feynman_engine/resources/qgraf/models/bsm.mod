% BSM simplified dark matter model for QGRAF
%
% Particles:
%   SM QED sector:  em (e-), ep (e+), mum (mu-), mup (mu+), A (photon)
%   Mediator:       Zp (Z' dark photon)
%   Dark matter:    chi (scalar DM), chia (scalar DM antiparticle)
%
% This model supports processes like:
%   e+ e- -> chi chi~  (DM pair production via Z' or photon)
%   chi chi~ -> e+ e-  (DM annihilation)
%   e+ e- -> e+ e-     (SM Bhabha scattering)

% Propagators: [field, dual-field, sign]
[em, ep, -]
[mum, mup, -]
[A, A, +]
[Zp, Zp, +]
[chi, chia, +]

% Vertices
% SM photon couplings
[em, ep, A]
[mum, mup, A]
% Z' couplings to SM fermions (dark photon couples like photon)
[em, ep, Zp]
[mum, mup, Zp]
% Z' coupling to dark matter
[chi, chia, Zp]
% Higgs portal: DM quartic self-coupling (optional contact term)
[chi, chia, chi, chia]
