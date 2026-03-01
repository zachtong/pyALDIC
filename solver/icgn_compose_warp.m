function P = icgn_compose_warp(P, DeltaP)
%ICGN_COMPOSE_WARP  Inverse compositional warp update for IC-GN solvers.
%   P = icgn_compose_warp(P, DeltaP) applies the IC-GN warp composition:
%     W(P) <- W(P) * W(DeltaP)^{-1}
%   where P = [F11-1, F21, F12, F22-1, Ux, Uy] (6-element vector).
%
%   Returns [] if det(DeltaP warp) == 0 (singular update).
%   Used by both icgn_solver.m (6-DOF) and icgn_subpb1.m (2-DOF).

    detDP = (1+DeltaP(1))*(1+DeltaP(4)) - DeltaP(2)*DeltaP(3);
    if detDP == 0
        P = [];  % signal failure to caller
        return
    end

    % Inverse of DeltaP warp
    iP1 = (-DeltaP(1) - DeltaP(1)*DeltaP(4) + DeltaP(2)*DeltaP(3)) / detDP;
    iP2 = -DeltaP(2) / detDP;
    iP3 = -DeltaP(3) / detDP;
    iP4 = (-DeltaP(4) - DeltaP(1)*DeltaP(4) + DeltaP(2)*DeltaP(3)) / detDP;
    iP5 = (-DeltaP(5) - DeltaP(4)*DeltaP(5) + DeltaP(3)*DeltaP(6)) / detDP;
    iP6 = (-DeltaP(6) - DeltaP(1)*DeltaP(6) + DeltaP(2)*DeltaP(5)) / detDP;

    % Compose: W(P) * W(DeltaP)^{-1}
    M = [1+P(1) P(3) P(5); P(2) 1+P(4) P(6); 0 0 1] * ...
        [1+iP1 iP3 iP5; iP2 1+iP4 iP6; 0 0 1];

    P = [M(1,1)-1; M(2,1); M(1,2); M(2,2)-1; M(1,3); M(2,3)];
end
